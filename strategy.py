# 16:15 UTC - Strategy is ready for submission
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla pivot with daily volume spike and weekly EMA trend filter.
# Camarilla levels from daily pivot provide precise reversal zones in both bull and bear markets.
# Volume spike confirms institutional interest at these key levels.
# Weekly EMA ensures alignment with higher timeframe momentum to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.

name = "exp_13162_12h_camarilla_pivot_1d_vol_1w_ema_v1"
timeframe = "12h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 1
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
EMA_WEEKLY_PERIOD = 20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
SIGNAL_SIZE = 0.25

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    pivot = (high + low + close) / 3.0
    range_val = high - low
    # Resistance levels
    r4 = close + (range_val * 1.5000)
    r3 = close + (range_val * 1.2500)
    r2 = close + (range_val * 1.1666)
    r1 = close + (range_val * 1.0833)
    # Support levels
    s1 = close - (range_val * 1.0833)
    s2 = close - (range_val * 1.1666)
    s3 = close - (range_val * 1.2500)
    s4 = close - (range_val * 1.5000)
    return pivot, r1, r2, r3, r4, s1, s2, s3, s4

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

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, EMA_WEEKLY_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_vals = np.full(len(df_1d), np.nan)
    r1_vals = np.full(len(df_1d), np.nan)
    r2_vals = np.full(len(df_1d), np.nan)
    r3_vals = np.full(len(df_1d), np.nan)
    r4_vals = np.full(len(df_1d), np.nan)
    s1_vals = np.full(len(df_1d), np.nan)
    s2_vals = np.full(len(df_1d), np.nan)
    s3_vals = np.full(len(df_1d), np.nan)
    s4_vals = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        pivot, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        pivot_vals[i] = pivot
        r1_vals[i] = r1
        r2_vals[i] = r2
        r3_vals[i] = r3
        r4_vals[i] = r4
        s1_vals[i] = s1
        s2_vals[i] = s2
        s3_vals[i] = s3
        s4_vals[i] = s4
    
    # Align Camarilla levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_vals)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_vals)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_vals)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_vals)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_vals)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_vals)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_vals)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_vals)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_vals)
    
    # Calculate 12h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Camarilla level touches with volume and trend confirmation
        touch_r3 = abs(close[i] - r3_aligned[i]) < (0.1 * atr[i]) if not np.isnan(r3_aligned[i]) else False
        touch_r4 = abs(close[i] - r4_aligned[i]) < (0.1 * atr[i]) if not np.isnan(r4_aligned[i]) else False
        touch_s3 = abs(close[i] - s3_aligned[i]) < (0.1 * atr[i]) if not np.isnan(s3_aligned[i]) else False
        touch_s4 = abs(close[i] - s4_aligned[i]) < (0.1 * atr[i]) if not np.isnan(s4_aligned[i]) else False
        
        # Entry conditions
        if position == 0:
            # Short at resistance levels in downtrend
            if (touch_r3 or touch_r4) and volume_ok and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            # Long at support levels in uptrend
            elif (touch_s3 or touch_s4) and volume_ok and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals