#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Camarilla pivot levels (H4/L4) for breakout entries,
# 1d EMA34 trend filter, and volume spike confirmation. Works in bull (long H4 breakouts in uptrend)
# and bear (short L4 breakdowns in downtrend) regimes. Target: 50-150 trades over 4 years (12-37/year).
# Size: 0.25.

name = "12h_Camarilla_H4L4_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots, EMA34, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Camarilla pivot levels (H4, L4) from previous day
    # Pivot = (H + L + C) / 3
    # H4 = C + ((H - L) * 1.1 / 2)
    # L4 = C - ((H - L) * 1.1 / 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    H4_1d = close_1d + ((high_1d - low_1d) * 1.1 / 2.0)
    L4_1d = close_1d - ((high_1d - low_1d) * 1.1 / 2.0)
    
    # Calculate 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume average (20-period) for volume spike confirmation
    volume_1d_series = pd.Series(volume_1d)
    vol_avg_20_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 12h timeframe
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # 12h ATR(14) for stoploss (not used in signal generation but for risk context)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need EMA34 and volume average to be ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(H4_1d_aligned[i]) or 
            np.isnan(L4_1d_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or
            np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction
        uptrend = close_1d_aligned[i] > ema34_1d_aligned[i]  # Using aligned close for consistency
        downtrend = close_1d_aligned[i] < ema34_1d_aligned[i]
        
        # Volume confirmation: current 12h volume > 1.5 * 1d average volume (scaled)
        # Approximate 1d volume equivalent for 12h bar: 1d volume / 2
        vol_equiv_12h = vol_avg_20_1d_aligned[i] / 2.0
        volume_spike = volume[i] > 1.5 * vol_equiv_12h
        
        # Breakout conditions
        long_breakout = close[i] > H4_1d_aligned[i]
        short_breakout = close[i] < L4_1d_aligned[i]
        
        # Entry logic
        long_entry = uptrend and long_breakout and volume_spike
        short_entry = downtrend and short_breakout and volume_spike
        
        # Exit logic: opposite Camarilla level touch (H3/L3) for faster reversion
        # H3 = C + ((H - L) * 1.1 / 4)
        # L3 = C - ((H - L) * 1.1 / 4)
        H3_1d = close_1d + ((high_1d - low_1d) * 1.1 / 4.0)
        L3_1d = close_1d - ((high_1d - low_1d) * 1.1 / 4.0)
        H3_1d_aligned = align_htf_to_ltf(prices, df_1d, H3_1d)
        L3_1d_aligned = align_htf_to_ltf(prices, df_1d, L3_1d)
        
        long_exit = close[i] < H3_1d_aligned[i]
        short_exit = close[i] > L3_1d_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals