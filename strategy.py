#!/usr/bin/env python3
name = "6h_48hDonchian_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

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
    
    # Load 1-day data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 48-period Donchian channel (8 bars = 48h = 2 days) on 6h timeframe
    donchian_high = pd.Series(high).rolling(window=8, min_periods=8).max().values
    donchian_low = pd.Series(low).rolling(window=8, min_periods=8).min().values
    
    # Daily EMA(34) trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: 24-period average (4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(8, 24, 34)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above 48h high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > donchian_high[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: break below 48h low with volume and daily downtrend
            elif close[i] < donchian_low[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: break below 48h low or trend reversal
            if close[i] < donchian_low[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: break above 48h high or trend reversal
            if close[i] > donchian_high[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 48-hour Donchian breakout with daily trend filter
# - 48h (8 bars) Donchian channel captures short-term momentum bursts
# - Breakout above 48h high with volume (>1.5x avg) in daily uptrend = long
# - Breakdown below 48h low with volume in daily downtrend = short
# - Daily EMA(34) filter ensures alignment with longer-term trend
# - Exit on opposite Donchian break or trend reversal
# - Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Position size 0.30 targets ~20-40 trades/year, avoiding fee drag
# - Simple, robust structure that avoids overfitting and works across regimes