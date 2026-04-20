# 6h_1d_Camarilla_R1S1_Breakout_Volume_Trend_v3
# Hypothesis: On 6h timeframe, price breaking above/below daily Camarilla R1/S1 with volume confirmation and 6h trend filter (EMA34) captures institutional breakout moves.
# Uses daily Camarilla levels for structure, volume for confirmation, and EMA34 on 6h for trend filter to avoid counter-trend trades.
# Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing to minimize fee drag.
# Works in bull/bear: trend filter ensures alignment with higher timeframe momentum, volume confirms institutional participation.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Camarilla_R1S1_Breakout_Volume_Trend_v3"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Camarilla Levels (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Camarilla calculation
    range_val = prev_high - prev_low
    close_prev = prev_close
    
    # Key levels: R1, S1, R2, S2
    r1 = close_prev + (range_val * 1.1 / 12)
    s1 = close_prev - (range_val * 1.1 / 12)
    r2 = close_prev + (range_val * 1.1 / 6)
    s2 = close_prev - (range_val * 1.1 / 6)
    
    # Align to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # === 6h EMA34 Trend Filter ===
    close_6h = prices['close'].values
    ema_34 = pd.Series(close_6h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 6h Volume Confirmation ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_34[i]
        vol_ratio_val = vol_ratio[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(vol_ratio_val) or 
            np.isnan(r1_val) or np.isnan(s1_val) or
            np.isnan(r2_val) or np.isnan(s2_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume and above EMA34 (uptrend)
            if close_val > r1_val and vol_ratio_val > 2.0 and close_val > ema_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume and below EMA34 (downtrend)
            elif close_val < s1_val and vol_ratio_val > 2.0 and close_val < ema_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Break below S1 or loss of trend
            if close_val < s1_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Break above R1 or loss of trend
            if close_val > r1_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals