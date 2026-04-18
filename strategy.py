# 12h_Camarilla_Pivot_R1_S1_Breakout_Volume_ATRFilter_v1
# Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as strong support/resistance.
# Breakouts with volume confirmation and ATR filter for volatility regime work in both bull/bear markets.
# Pivot levels derived from prior day's range provide objective levels that work across regimes.
# Volume surge confirms institutional interest. ATR filter avoids choppy markets.
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
# Strategy uses discrete position sizing (0.25) to reduce churn.

#!/usr/bin/env python3
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.full_like(close_1d, np.nan)
    camarilla_s1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        H = high_1d[i]
        L = low_1d[i]
        C = close_1d[i]
        camarilla_r1[i] = C + (H - L) * 1.1 / 12
        camarilla_s1[i] = C - (H - L) * 1.1 / 12
    
    # Calculate 14-day ATR for volatility filter
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for volume spike detection
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all daily data to 12h timeframe
    camarilla_r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    atr_14_12h = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ma_20_12h = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 12h volume spike (current volume > 1.8x 20-period average)
    vol_ma_12h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_12h[i]) or np.isnan(camarilla_s1_12h[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 20-period average (avoid chop)
        vol_filter = atr_14_12h[i] > vol_ma_20_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and volatility filter
            if close[i] > camarilla_r1_12h[i] and volume_spike[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and volatility filter
            elif close[i] < camarilla_s1_12h[i] and volume_spike[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 OR volatility dies
            if close[i] < camarilla_s1_12h[i] or not vol_filter:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 OR volatility dies
            if close[i] > camarilla_r1_12h[i] or not vol_filter:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_R1_S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0