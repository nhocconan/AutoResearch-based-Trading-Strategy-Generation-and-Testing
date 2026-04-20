# 6h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1
# 6h timeframe with 1d Camarilla pivot breakout + volume confirmation + ATR filter
# Designed for balanced performance in both bull and bear markets
# Breakouts are filtered by volume (avoid fakeouts) and ATR (volatility regime)
# Targets 50-150 trades over 4 years = 12-37/year to minimize fee drag
# Uses discrete position sizing (0.25) to reduce churn

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # R2 = C + (H-L) * 1.1/6
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # S2 = C - (H-L) * 1.1/6
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    
    # Previous day's values (shifted by 1 to avoid look-ahead)
    prev_high = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low = np.concatenate([[np.nan], low_1d[:-1]])
    prev_close = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Calculate pivot levels
    hl_range = prev_high - prev_low
    r4 = prev_close + hl_range * 1.1 / 2
    r3 = prev_close + hl_range * 1.1 / 4
    r2 = prev_close + hl_range * 1.1 / 6
    r1 = prev_close + hl_range * 1.1 / 12
    s1 = prev_close - hl_range * 1.1 / 12
    s2 = prev_close - hl_range * 1.1 / 6
    s3 = prev_close - hl_range * 1.1 / 4
    s4 = prev_close - hl_range * 1.1 / 2
    
    # Daily ATR(14) for volatility filter
    tr1 = prev_high[1:] - prev_low[1:]
    tr2 = np.abs(prev_high[1:] - prev_close[:-1])
    tr3 = np.abs(prev_low[1:] - prev_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume moving average (20-period)
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily indicators to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    # 6h price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in daily indicators
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current volume must be above 20-day average
        vol_filter = volume[i] > volume_ma_20_1d_aligned[i]
        
        # ATR filter: only trade when volatility is below 80th percentile (avoid extreme volatility)
        atr_threshold = np.nanpercentile(atr_14_1d_aligned[:i+1], 80) if i >= 50 else atr_14_1d_aligned[i]
        vol_regime = atr_14_1d_aligned[i] < atr_threshold
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume and volatility filter
            if price > r1_val and vol_filter and vol_regime:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S1 with volume and volatility filter
            elif price < s1_val and vol_filter and vol_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reversal) or extreme volatility
            if price < s1_val or atr_14_1d_aligned[i] > np.nanpercentile(atr_14_1d_aligned[:i+1], 90):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reversal) or extreme volatility
            if price > r1_val or atr_14_1d_aligned[i] > np.nanpercentile(atr_14_1d_aligned[:i+1], 90):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Camarilla_R1_S1_Breakout_Volume_ATRFilter_V1"
timeframe = "6h"
leverage = 1.0