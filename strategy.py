#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 12h trend filter and volume confirmation.
# Uses Bollinger Band width percentile to detect low volatility squeezes, then breaks out in direction
# of 12h trend with volume confirmation. Works in both bull and bear markets by capturing
# volatility expansion after consolidation periods. Target: 80-180 total trades over 4 years (20-45/year).

name = "exp_13459_6h_bb_squeeze_12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
BB_PERIOD = 20
BB_STD = 2.0
BBW_PERIOD = 50  # for percentile calculation
EMA_PERIOD = 21
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_bollinger_bands(close, period, std):
    """Calculate Bollinger Bands"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean()
    std_dev = pd.Series(close).rolling(window=period, min_periods=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper.values, lower.values, sma.values

def calculate_bb_width_percentile(upper, lower, sma, period):
    """Calculate Bollinger Band width and its percentile"""
    bb_width = (upper - lower) / sma
    # Calculate percentile rank over lookback period
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=period, min_periods=period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    return bb_width_pct

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, EMA_PERIOD)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Bollinger Bands
    bb_upper, bb_lower, bb_middle = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    
    # Bollinger Band width percentile (squeeze detector)
    bbw_percentile = calculate_bb_width_percentile(bb_upper, bb_lower, bb_middle, BBW_PERIOD)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(BBW_PERIOD, EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(bbw_percentile[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # Conditions
        bb_squeeze = bbw_percentile[i] < 0.2  # Low volatility squeeze (<20th percentile)
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Breakout conditions
        breakout_up = bb_squeeze and volume_ok and uptrend and (close[i] > bb_upper[i])
        breakout_down = bb_squeeze and volume_ok and downtrend and (close[i] < bb_lower[i])
        
        # Generate signals
        if position == 0:
            if breakout_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_down:
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