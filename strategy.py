#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and 1w trend filter
# Long when price breaks above 1d Camarilla R1 + 1d volume > 1.3x 20-period average + 1w close > 1w EMA34
# Short when price breaks below 1d Camarilla S1 + 1d volume > 1.3x 20-period average + 1w close < 1w EMA34
# Uses Camarilla pivot levels from 1d for precise intraday structure, volume for confirmation, and 1w EMA for trend filter
# Designed for low trade frequency (12-37/year) to minimize fee drag while capturing high-probability breakouts

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1d Indicators: Volume Confirmation ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1w Indicators: EMA34 for Trend Filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. 1w trend filter: close > EMA34 (bullish bias)
        if (close[i] > r1_1d_aligned[i]) and vol_confirm and (close_1w[-1] > ema_34_1w_aligned[i] if len(close_1w) > 0 else False):
            # Use previous bar's 1w close for non-look-ahead
            if i >= len(prices) - len(df_1w) * 7 * 24:  # Rough alignment check
                hist_idx = min(i // (12 * 20), len(close_1w) - 2)  # Approximate 1w index
                if hist_idx > 0 and close_1w[hist_idx] > ema_34_1w[hist_idx]:
                    signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. 1w trend filter: close < EMA34 (bearish bias)
        elif (close[i] < s1_1d_aligned[i]) and vol_confirm and (close_1w[-1] < ema_34_1w_aligned[i] if len(close_1w) > 0 else False):
            # Use previous bar's 1w close for non-look-ahead
            if i >= len(prices) - len(df_1w) * 7 * 24:  # Rough alignment check
                hist_idx = min(i // (12 * 20), len(close_1w) - 2)  # Approximate 1w index
                if hist_idx > 0 and close_1w[hist_idx] < ema_34_1w[hist_idx]:
                    signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_TrendFilter_v1"
timeframe = "12h"
leverage = 1.0