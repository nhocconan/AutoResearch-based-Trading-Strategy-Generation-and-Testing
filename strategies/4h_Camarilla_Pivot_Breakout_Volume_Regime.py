# 4h_Camarilla_Pivot_Breakout_Volume_Regime
# Hypothesis: Camarilla pivot levels act as institutional support/resistance.
# Breakout above R3 or below S3 with volume surge indicates institutional participation.
# Combines with 12h EMA trend filter and daily ATR percentile regime filter to avoid whipsaws.
# Works in bull/bear: breakouts capture momentum, regime filter avoids chop, volume confirms validity.
# Target: 20-40 trades/year per symbol.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Camarilla pivots and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Load 12h data ONCE for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === Daily Camarilla Pivot Levels (from previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels using previous day's data
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: R3/S3 are most significant for breakouts
    r3 = close_1d + (range_hl * 1.1 / 2)
    s3 = close_1d - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # === Daily ATR percentile for regime filter ===
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_percentile = pd.Series(atr_14).rolling(window=100, min_periods=14).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        r3_level = r3_aligned[i]
        s3_level = s3_aligned[i]
        ema_trend = ema_50_12h_aligned[i]
        atr_percentile_val = atr_percentile_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above R3 (resistance) with volume in favorable regime
            if (price_close > r3_level and  # Breakout above R3
                vol_ratio_val > 1.8 and     # Volume surge
                atr_percentile_val < 40):   # Low-moderate volatility (avoid extreme chop)
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S3 (support) with volume in favorable regime
            elif (price_close < s3_level and   # Breakdown below S3
                  vol_ratio_val > 1.8 and      # Volume surge
                  atr_percentile_val < 40):    # Low-moderate volatility
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: reverse breakdown/breakout or volatility expansion
            if position == 1 and (price_close < s3_level or atr_percentile_val > 80):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > r3_level or atr_percentile_val > 80):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_Pivot_Breakout_Volume_Regime"
timeframe = "4h"
leverage = 1.0