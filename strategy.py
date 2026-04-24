#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 Breakout with 4h EMA34 Trend Filter and Volume Spike.
- Primary timeframe: 1h for execution, HTF: 4h for EMA34 trend filter and 1d for Camarilla levels.
- Entry: Price breaks above Camarilla H3 (long) or below L3 (short) on 1h close, with volume > 1.8x 20-period volume MA.
- Direction filter: only long when 1h close > 4h EMA34, only short when 1h close < 4h EMA34.
- Camarilla levels from 1d provide strong intraday support/resistance; 4h EMA34 filters for trend alignment.
- Volume confirmation reduces false breakouts.
- Exit: Price returns to Camarilla Pivot Point (PP) or trend filter reversal.
- Discrete signal size: 0.20 to balance return and drawdown control.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
- Session filter: 08-20 UTC to reduce noise trades.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA34 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 1d Camarilla levels (based on previous 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift OHLC by 1 to use previous day's data (avoid look-ahead)
    high_1d_shifted = np.roll(high_1d, 1)
    low_1d_shifted = np.roll(low_1d, 1)
    close_1d_shifted = np.roll(close_1d, 1)
    # First bar: use same day's data (no prior day available)
    high_1d_shifted[0] = high_1d[0]
    low_1d_shifted[0] = low_1d[0]
    close_1d_shifted[0] = close_1d[0]
    
    # Camarilla calculations: based on previous day's range
    rng = high_1d_shifted - low_1d_shifted
    camarilla_pp = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3
    camarilla_h3 = camarilla_pp + 1.1 * (high_1d_shifted - low_1d_shifted) / 2
    camarilla_l3 = camarilla_pp - 1.1 * (high_1d_shifted - low_1d_shifted) / 2
    
    # Align Camarilla levels to 1h timeframe (completed 1d bar only)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # Need 4h EMA34, volume MA(20), plus 1 for Camarilla shift
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(volume_spike[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla H3 with volume spike AND uptrend (close > 4h EMA34)
            if (close[i] > camarilla_h3_aligned[i] and volume_spike[i] and 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below Camarilla L3 with volume spike AND downtrend (close < 4h EMA34)
            elif (close[i] < camarilla_l3_aligned[i] and volume_spike[i] and 
                  close[i] < ema_34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Price returns to Camarilla Pivot Point or trend reversal
            if (close[i] < camarilla_pp_aligned[i] or close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Price returns to Camarilla Pivot Point or trend reversal
            if (close[i] > camarilla_pp_aligned[i] or close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0