# 6H_WickReversal_SupportResistance_VolumeFilter
# Hypothesis: 6H wick rejection at daily support/resistance with volume confirmation
# captures institutional order flow exhaustion. Works in bull/bear by fading extremes.
# Target: 20-40 trades/year via strict 3-condition confluence (wick + level + volume)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Daily data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot-based support/resistance (more robust than raw highs/lows)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d      # Resistance 1
    s1_1d = 2 * pivot_1d - high_1d     # Support 1
    r2_1d = pivot_1d + (high_1d - low_1d)  # Resistance 2
    s2_1d = pivot_1d - (high_1d - low_1d)  # Support 2
    
    # Align daily levels to 6H timeframe
    s1_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2_1d)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2_1d)
    
    # 6H ATR for volatility filter and stop
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 20-period volume MA surge
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20  # Volume surge threshold
    
    # Wick rejection detection: long tail, small body
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    total_range = high - low
    
    # Avoid division by zero
    valid_range = total_range > 0
    upper_wick_ratio = np.where(valid_range, upper_wick / total_range, 0)
    lower_wick_ratio = np.where(valid_range, lower_wick / total_range, 0)
    body_ratio = np.where(valid_range, body_size / total_range, 1)
    
    # Strong wick rejection: long tail (>60%), small body (<30%)
    strong_lower_wick = lower_wick_ratio > 0.6
    strong_upper_wick = upper_wick_ratio > 0.6
    small_body = body_ratio < 0.3
    
    signals = np.zeros(n)
    position = 0  # 0=flat, 1=long, -1=short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(s1_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(s2_6h[i]) or 
            np.isnan(r2_6h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: bullish wick rejection at support with volume surge
            # Price near S1/S2, strong lower wick, small body, volume surge
            near_support = (low[i] <= s1_6h[i] * 1.005) or (low[i] <= s2_6h[i] * 1.005)
            if near_support and strong_lower_wick[i] and small_body[i] and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish wick rejection at resistance with volume surge
            # Price near R1/R2, strong upper wick, small body, volume surge
            elif (high[i] >= r1_6h[i] * 0.995) or (high[i] >= r2_6h[i] * 0.995):
                if strong_upper_wick[i] and small_body[i] and vol_surge[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: opposite wick rejection or volatility collapse
            if position == 1:  # Long position
                # Exit on bearish rejection at resistance or volatility drop
                near_resistance = (high[i] >= r1_6h[i] * 0.995) or (high[i] >= r2_6h[i] * 0.995)
                if near_resistance and strong_upper_wick[i] and small_body[i]:
                    signals[i] = 0.0
                    position = 0
                elif atr[i] < 0.4 * atr[i-1]:  # Volatility collapse filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short)
                # Exit on bullish rejection at support or volatility drop
                near_support = (low[i] <= s1_6h[i] * 1.005) or (low[i] <= s2_6h[i] * 1.005)
                if near_support and strong_lower_wick[i] and small_body[i]:
                    signals[i] = 0.0
                    position = 0
                elif atr[i] < 0.4 * atr[i-1]:  # Volatility collapse filter
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_WickReversal_SupportResistance_VolumeFilter"
timeframe = "6h"
leverage = 1.0