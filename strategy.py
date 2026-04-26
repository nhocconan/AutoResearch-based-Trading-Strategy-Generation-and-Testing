#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1
Hypothesis: Daily Camarilla R3/S3 breakout with 1-week trend filter and volume confirmation.
- Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
- Long when price breaks above R3 AND 1w uptrend AND volume spike
- Short when price breaks below S3 AND 1w downtrend AND volume spike
- Camarilla levels act as intraday support/resistance derived from prior day range
- 1-week EMA34 trend filter reduces whipsaw in choppy markets
- Volume spike (1.8x 20-period average) confirms institutional participation
- Designed for low frequency with proven edge on BTC/ETH from pivot breakouts in trending markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels for each day (using prior day's OHLC)
    # R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = (H+L+O)/3 (typical price)
    
    # Shift by 1 to use prior day's data for today's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(prices['open'].values, 1)
    
    # First value is invalid due to shift
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    prev_open[0] = np.nan
    
    # Typical price (pivot point)
    typical_price = (prev_high + prev_low + prev_close) / 3
    # Range
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    R3 = typical_price + range_hl * 1.1 / 4
    S3 = typical_price - range_hl * 1.1 / 4
    
    # Calculate 1-week EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate volume spike (20-period volume average on 1d)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)  # Volume at least 1.8x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for shift)
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with volume confirmation and trend filter
        if position == 0:
            # Long: Price breaks above R3 AND 1w uptrend AND volume spike
            if close[i] > R3[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 AND 1w downtrend AND volume spike
            elif close[i] < S3[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls back below R3 OR 1w trend turns down
            if close[i] < R3[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises back above S3 OR 1w trend turns up
            if close[i] > S3[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0