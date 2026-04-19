#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout with daily volume confirmation and ATR-based stop-loss.
# Breakouts above/below 20-period Donchian channels capture trend continuations.
# Volume confirmation filters false breakouts. Works in bull/bear as breakouts capture trends
# and reversals at channel boundaries provide mean-reversion opportunities.
# Target: 50-150 trades over 4 years to balance signal quality and fee drag.

name = "4h_Donchian20_VolumeConfirm_ATRStop"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume for confirmation
    vol_ma_20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    # Calculate 4-hour Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4-hour ATR for stop-loss
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_24h = volume_1d[i] if i < len(volume_1d) else volume_1d[-1]
        vol_ma = vol_ma_20d_aligned[i]
        atr_val = atr[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        
        volume_confirmed = vol_24h > 1.5 * vol_ma
        
        if position == 0:
            # Long: Break above upper Donchian band with volume confirmation
            if price > upper and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian band with volume confirmation
            elif price < lower and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below lower band or ATR stop
            if price < lower or price < (high[i] - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above upper band or ATR stop
            if price > upper or price > (low[i] + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals