# 12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Uses daily trend filter (EMA34) with daily Camarilla R3/S3 breakouts on 12h timeframe.
# Daily trend reduces false breakouts in choppy markets, while daily pivot levels provide
# precise entry/exit points. Volume confirmation ensures breakout strength. Designed for
# low trade frequency (12-37/year) to minimize fee drag in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily EMA34 for trend filter
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate typical price and range from previous day
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    # Camarilla R3 and S3 levels
    R3 = typical_price + (range_hl * 1.2500)
    S3 = typical_price - (range_hl * 1.2500)
    # Align daily levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    
    # Volume confirmation (24-period average on 12h = ~12 days)
    vol_ma_period = 24
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, vol_ma_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 34) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average (stricter for fewer trades)
        volume_confirm = volume[i] > 1.8 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R3 with volume, above daily EMA34 (uptrend)
            if close[i] > R3_aligned[i] and volume_confirm and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, below daily EMA34 (downtrend)
            elif close[i] < S3_aligned[i] and volume_confirm and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 or breaks below daily EMA34
            if close[i] < S3_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 or breaks above daily EMA34
            if close[i] > R3_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals