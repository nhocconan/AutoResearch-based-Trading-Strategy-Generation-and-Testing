# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: 12h Camarilla R1/S1 breakout with daily EMA trend filter and volume confirmation.
# Uses daily Camarilla levels (R1/S1) for entry, daily EMA50 for trend direction, and volume spike for confirmation.
# Designed for 12h timeframe: expects 15-35 trades/year per symbol.
# Works in bull/bear: EMA trend adapts to daily structure; volume avoids false breaks.
# Stops when price crosses back through the opposite Camarilla level (S1 for long, R1 for short).
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
    
    # Get daily data for Camarilla and EMA
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    R1_d = close_d + (high_d - low_d) * 1.1 / 12
    S1_d = close_d - (high_d - low_d) * 1.1 / 12
    
    # Align daily Camarilla to 12h
    R1_d_aligned = align_htf_to_ltf(prices, df_d, R1_d)
    S1_d_aligned = align_htf_to_ltf(prices, df_d, S1_d)
    
    # Calculate daily EMA50 for trend filter
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_d_aligned = align_htf_to_ltf(prices, df_d, ema50_d)
    
    # Calculate 20-period volume average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(vol_period, 50) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(R1_d_aligned[i]) or np.isnan(S1_d_aligned[i]) or 
            np.isnan(ema50_d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        if position == 0:
            # Long: Price breaks above R1 with volume AND above daily EMA50
            if price > R1_d_aligned[i] and vol_ratio > 2.0 and price > ema50_d_aligned[i]:
                signals[i] = size
                position = 1
            # Short: Price breaks below S1 with volume AND below daily EMA50
            elif price < S1_d_aligned[i] and vol_ratio > 2.0 and price < ema50_d_aligned[i]:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below S1 (opposite level)
            if price < S1_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above R1 (opposite level)
            if price > R1_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0