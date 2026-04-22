# 2. Hypothesis: A 4h strategy using 1-day ATR-based volatility filtering on Camarilla pivot breakouts (S3/R3) with volume surge confirmation.
# This combines the edge of price reacting at statistically significant intraday levels (Camarilla S3/R3),
# filtered by elevated volatility (ATR surge) to avoid chop, and volume to confirm institutional interest.
# Works in bull/bear because breakouts capture momentum in any regime, while volatility filter avoids false signals in low-volatility environments.
# Target: 20-40 trades/year (~80-160 over 4 years) to stay within fee-efficient range.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot points (S3 and R3 only - the key breakout levels)
    range_1d = high_1d - low_1d
    close_prev = close_1d
    s3_1d = close_prev - (range_1d * 3.0 / 6)  # S3 = C - (H-L)*3/6
    r3_1d = close_prev + (range_1d * 3.0 / 6)  # R3 = C + (H-L)*3/6
    
    # Align daily S3/R3 to 4h timeframe
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    
    # 4h ATR for volatility filter (14-period) - using Wilder's smoothing
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_raw = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # Wilder's ATR smoothing: ATR[t] = (ATR[t-1] * 13 + TR[t]) / 14
    atr = np.full_like(atr_raw, np.nan)
    atr[13] = atr_raw[13]  # Seed with first 14-period average
    for i in range(14, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # ATR ratio: current ATR vs 50-period average to detect volatility surge
    atr_ma50 = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / atr_ma50
    
    # Volume filter (20-period MA)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20  # Slightly lower threshold for more signals
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after warmup period for ATR and MA
        # Skip if data not ready
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(atr_ma50[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volatility surge and volume confirmation
            if close[i] > r3_1d_aligned[i] and atr_ratio[i] > 1.3 and vol_surge[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with volatility surge and volume confirmation
            elif close[i] < s3_1d_aligned[i] and atr_ratio[i] > 1.3 and vol_surge[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the midpoint between S3 and R3 (mean reversion to value area)
            midpoint = (s3_1d_aligned[i] + r3_1d_aligned[i]) / 2
            if position == 1:
                if close[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_S3R3_Breakout_ATR_Volume_Surge"
timeframe = "4h"
leverage = 1.0