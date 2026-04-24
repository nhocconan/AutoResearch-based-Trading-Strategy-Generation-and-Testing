#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 Breakout with 1d EMA34 Trend Filter and Volume Spike.
- Primary timeframe: 4h for execution, HTF: 1d for EMA34 trend filter.
- Entry: Price breaks above Camarilla R3 (long) or below S3 (short) on 4h close, with volume > 1.8x 20-period volume MA.
- Direction filter: only long when 4h close > 1d EMA34, only short when 4h close < 1d EMA34.
- Camarilla levels from 1d provide strong intraday support/resistance; EMA34 filters for trend alignment.
- Volume confirmation reduces false breakouts.
- Exit: Price returns to Camarilla Pivot Point (PP) or trend filter reversal.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla levels (based on previous 1d OHLC)
    # We need to shift by 1 to avoid look-ahead: use previous day's OHLC for today's levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shifted = np.roll(close_1d, 1)  # previous day's close
    close_1d_shifted[0] = close_1d[0]  # first bar uses same day (no prior)
    
    # Camarilla calculations: based on previous day's range
    rng = high_1d - low_1d
    camarilla_pp = (high_1d + low_1d + close_1d_shifted) / 3
    camarilla_r3 = camarilla_pp + 1.1 * (high_1d - low_1d) / 2
    camarilla_s3 = camarilla_pp - 1.1 * (high_1d - low_1d) / 2
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20) + 1  # Need 1d EMA34, volume MA(20), plus 1 for Camarilla shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Camarilla R3 with volume spike AND uptrend (close > 1d EMA34)
            if (close[i] > camarilla_r3_aligned[i] and volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 with volume spike AND downtrend (close < 1d EMA34)
            elif (close[i] < camarilla_s3_aligned[i] and volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Camarilla Pivot Point or trend reversal
            if (close[i] < camarilla_pp_aligned[i] or close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Camarilla Pivot Point or trend reversal
            if (close[i] > camarilla_pp_aligned[i] or close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0