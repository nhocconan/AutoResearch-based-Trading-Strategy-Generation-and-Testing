# 1d/1w Trend Following with Weekly EMA Trend Filter and Daily Momentum Confirmation
# Uses weekly EMA to establish trend direction, daily EMA for momentum confirmation
# Volume filter ensures trades occur with conviction
# Aims for 50-150 total trades over 4 years (12-38/year) to minimize fee drag
# Works in bull markets (trend following) and bear markets (counter-trend bounces at key levels)

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_12890_1d_weekly_ema_trend_daily_momentum_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
WEEKLY_EMA_PERIOD = 21
DAILY_EMA_FAST = 9
DAILY_EMA_SLOW = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_ema(close, period):
    """Calculate EMA with proper Wilder's smoothing"""
    return pd.Series(close).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend direction
    weekly_close = df_weekly['close'].values
    weekly_ema = calculate_ema(weekly_close, WEEKLY_EMA_PERIOD)
    
    # Align weekly EMA to daily timeframe
    weekly_ema_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema)
    
    # Calculate daily indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    daily_ema_fast = calculate_ema(close, DAILY_EMA_FAST)
    daily_ema_slow = calculate_ema(close, DAILY_EMA_SLOW)
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEKLY_EMA_PERIOD, DAILY_EMA_SLOW, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(weekly_ema_aligned[i]):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Daily EMA crossover for momentum
        ema_bullish = daily_ema_fast[i] > daily_ema_slow[i]
        ema_bearish = daily_ema_fast[i] < daily_ema_slow[i]
        
        # Trend filter: weekly EMA slope
        if i > 1:
            weekly_ema_rising = weekly_ema_aligned[i] > weekly_ema_aligned[i-1]
            weekly_ema_falling = weekly_ema_aligned[i] < weekly_ema_aligned[i-1]
        else:
            weekly_ema_rising = False
            weekly_ema_falling = False
        
        # Generate signals
        if position == 0:
            # Long: weekly uptrend + daily bullish momentum + volume
            if weekly_ema_rising and ema_bullish and volume_ok:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            # Short: weekly downtrend + daily bearish momentum + volume
            elif weekly_ema_falling and ema_bearish and volume_ok:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Stay long until stop
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Stay short until stop
            signals[i] = -SIGNAL_SIZE
    
    return signals