# 1d_Camarilla_Pivot_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot breakouts on 1d with 1w trend filter and volume confirmation
# Works in bull and bear markets by following 1w trend and using volatility-based entries from pivot levels
# Target: 15-25 trades/year per symbol with strict entry conditions to minimize fee drag

name = "1d_Camarilla_Pivot_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Using previous day's high, low, close
    phigh = np.roll(high, 1)
    plow = np.roll(low, 1)
    pclose = np.roll(close, 1)
    phigh[0] = high[0]  # first bar uses current bar
    plow[0] = low[0]
    pclose[0] = close[0]
    
    # Calculate pivot point and ranges
    pivot = (phigh + plow + pclose) / 3
    range_val = phigh - plow
    
    # Camarilla levels
    R4 = pclose + range_val * 1.1 / 2
    R3 = pclose + range_val * 1.1 / 4
    R2 = pclose + range_val * 1.1 / 6
    R1 = pclose + range_val * 1.1 / 12
    S1 = pclose - range_val * 1.1 / 12
    S2 = pclose - range_val * 1.1 / 6
    S3 = pclose - range_val * 1.1 / 4
    S4 = pclose - range_val * 1.1 / 2
    
    # Calculate 1w EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_20_1w[i-1]
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate volume SMA(10)
    vol_sma = np.full(n, np.nan)
    for i in range(10, n):
        vol_sma[i] = np.mean(volume[i-10:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above R3 with uptrend and volume confirmation
            if close[i] > R3[i] and close[i] > ema_20_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend and volume confirmation
            elif close[i] < S3[i] and close[i] < ema_20_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Close crosses back below pivot or R1
            if close[i] < pivot[i] or close[i] < R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Close crosses back above pivot or S1
            if close[i] > pivot[i] or close[i] > S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals