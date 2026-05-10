# %%
#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R3_S3_Breakout_12hEMA50_Trend_Volume
Hypothesis: Uses daily Camarilla pivot levels (R3/S3) as breakout levels on 4h timeframe,
with 12h EMA50 trend filter and volume confirmation to avoid false breakouts.
Designed for low trade frequency (20-30/year) to minimize fee drag while capturing
strong momentum moves in both bull and bear markets. Camarilla levels work well
as they represent key support/resistance where price often accelerates after breaking.
"""

name = "4h_Camarilla_Pivot_R3_S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500)
    # S3 = C - ((H-L) * 1.2500), S4 = C - ((H-L) * 1.5000)
    # where C = (H+L+C)/3 (typical price)
    
    # We'll use previous day's data to avoid look-ahead
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    # Camarilla levels (using previous day's data)
    R3 = typical_price + (range_hl * 1.2500)
    S3 = typical_price - (range_hl * 1.2500)
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3.values)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3.values)
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation (20-period average)
    vol_ma_period = 20
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    
    vol_ma = mean_arr(volume, vol_ma_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R3 with volume, above 12h EMA50
            if close[i] > R3_aligned[i] and volume_confirm and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume, below 12h EMA50
            elif close[i] < S3_aligned[i] and volume_confirm and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 or breaks below 12h EMA50
            if close[i] < S3_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 or breaks above 12h EMA50
            if close[i] > R3_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# %%