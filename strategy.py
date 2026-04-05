# 1. Hypothesis: 4h timeframe balances responsiveness and trade frequency to avoid fee drag. Using 12h trend (EMA50) ensures alignment with medium-term momentum, while 1d Donchian(20) breakout captures significant moves. Volume confirmation filters for institutional participation, and ATR-based stops manage risk. Targets 75-200 trades over 4 years (19-50/year) for statistical validity while minimizing fee impact.

#!/usr/bin/env python3

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8743_4h_donchian20_12h_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20  # 1d Donchian for breakout
TREND_PERIOD = 50     # 12h EMA for trend filter
VOLUME_MA_PERIOD = 20 # 4h volume MA for confirmation
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25    # Position size as fraction of capital
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

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
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=TREND_PERIOD, adjust=False, min_periods=TREND_PERIOD).mean().values
    
    # Price relative to 12h EMA: above = bullish bias, below = bearish bias
    price_vs_ema = np.where(close_12h > ema_12h, 1, 
                     np.where(close_12h < ema_12h, -1, 0))  # 1=bullish, -1=bearish, 0=at EMA
    price_vs_ema_aligned = align_htf_to_ltf(prices, df_12h, price_vs_ema)
    
    # Calculate 1d Donchian channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = pd.Series(low_1d).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Calculate LTF indicators (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(TREND_PERIOD, DONCHIAN_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_ema_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        
        # Determine market bias from 12h EMA
        bull_bias = price_vs_ema_aligned[i] == 1   # 12h price above EMA50
        bear_bias = price_vs_ema_aligned[i] == -1  # 12h price below EMA50
        
        # 1d Donchian breakout conditions (using previous day's levels)
        long_breakout = close[i] > donchian_high[i-1]  # Break above previous day's high
        short_breakout = close[i] < donchian_low[i-1]  # Break below previous day's low
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = bull_bias and long_breakout and volume_confirmed
        short_entry = bear_bias and short_breakout and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals