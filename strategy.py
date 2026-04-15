#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 12h EMA trend filter + volume confirmation
# Long when Williams %R < -80 (oversold) + price > 12h EMA50 (uptrend) + volume > 1.3x avg
# Short when Williams %R > -20 (overbought) + price < 12h EMA50 (downtrend) + volume > 1.3x avg
# Uses discrete position sizing (0.25) to minimize fee churn
# Williams %R identifies exhaustion points in ranging markets, EMA filter ensures trend alignment
# Designed for low trade frequency (15-30/year) to avoid fee drag while capturing mean reversion within trends

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h and 12h HTF data once before loop
    df_6h = get_htf_data(prices, '6h')
    df_12h = get_htf_data(prices, '12h')
    if len(df_6h < 50) or len(df_12h < 50):
        return np.zeros(n)
    
    # === 6h Indicators: Williams %R (14) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high_14 - close) / (highest_high_14 - lowest_low_14))
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # === 12h Indicators: EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Williams %R indicates oversold (< -80)
        # 2. Price above 12h EMA50 (uptrend filter)
        # 3. Volume confirmation
        if (williams_r[i] < -80) and (close[i] > ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Williams %R indicates overbought (> -20)
        # 2. Price below 12h EMA50 (downtrend filter)
        # 3. Volume confirmation
        elif (williams_r[i] > -20) and (close[i] < ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_WilliamsR_12hEMA50_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0