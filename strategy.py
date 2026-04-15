#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly ATR(14) for volatility regime filter
    tr1 = weekly_high - weekly_low
    tr2 = np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]]))
    tr3 = np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    weekly_atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate weekly ATR percentile (50-period lookback) for regime detection
    atr_percentile = pd.Series(weekly_atr).rolling(window=50, min_periods=30).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1w, atr_percentile)
    
    # Calculate 6h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr_percentile_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in low volatility environments (weekly ATR percentile < 0.4)
        # This avoids high volatility chop and focuses on mean reversion in calm markets
        if atr_percentile_aligned[i] >= 0.4:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: 6h price touches or breaks below lower Donchian band with volume confirmation
        # Short: 6h price touches or breaks above upper Donchian band with volume confirmation
        # Discrete position sizing: 0.25
        
        # Long conditions: price at or below lower Donchian band (oversold bounce)
        if (low[i] <= lowest_low[i] and            # Price touches/below lower Donchian
            volume_ratio[i] > 1.5):                # Volume confirmation (strong interest)
            signals[i] = 0.25
            
        # Short conditions: price at or above upper Donchian band (overbought rejection)
        elif (high[i] >= highest_high[i] and       # Price touches/above upper Donchian
              volume_ratio[i] > 1.5):              # Volume confirmation (strong interest)
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian_Touch_Volume_LowVol_Regime"
timeframe = "6h"
leverage = 1.0