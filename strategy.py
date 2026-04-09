#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h volume confirmation and ATR regime filter
# Uses 20-period Donchian channels on 6h for breakout signals
# Volume confirmation: 12h volume > 1.3x 24-period average (~12 days) to ensure institutional participation
# ATR regime filter: Only trade when 12h ATR(14) is between 30th-70th percentile (avoid extremes)
# Position size 0.25 to balance profit potential and drawdown control
# Target: 12-30 trades/year per symbol (48-120 total over 4 years) to minimize fee drag
# Works in bull/bear: Donchian provides structure, volume confirms strength, ATR filter avoids chop

name = "6h_12h_donchian_vol_atr_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for volume and ATR
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ATR (14-period)
    tr_12h = np.zeros(len(df_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(df_12h)):
        tr0 = high_12h[i] - low_12h[i]
        tr1 = abs(high_12h[i] - close_12h[i-1])
        tr2 = abs(low_12h[i] - close_12h[i-1])
        tr_12h[i] = max(tr0, tr1, tr2)
    
    atr_12h = np.zeros(len(df_12h))
    atr_12h[0] = tr_12h[0]
    for i in range(1, len(df_12h)):
        atr_12h[i] = (atr_12h[i-1] * 13 + tr_12h[i]) / 14
    
    # ATR percentile rank (100-period lookback ~ 50 days)
    atr_rank_12h = np.zeros(len(df_12h))
    for i in range(100, len(df_12h)):
        window = atr_12h[i-100:i]
        atr_rank_12h[i] = np.sum(window < atr_12h[i]) / len(window) * 100
    
    # Calculate 24-period volume average on 12h (~12 days)
    vol_ma_24 = np.full(len(df_12h), np.nan)
    vol_sum = 0.0
    for i in range(len(df_12h)):
        vol_sum += vol_12h[i]
        if i >= 24:
            vol_sum -= vol_12h[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    # Align 12h data to 6h timeframe (only use completed 12h bars)
    vol_ma_24_6h = align_htf_to_ltf(prices, df_12h, vol_ma_24)
    atr_rank_12h_6h = align_htf_to_ltf(prices, df_12h, atr_rank_12h)
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            start_idx = i - 19
            donchian_high[i] = np.max(high[start_idx:i+1])
            donchian_low[i] = np.min(low[start_idx:i+1])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup periods
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_24_6h[i]) or 
            np.isnan(atr_rank_12h_6h[i])):
            signals[i] = 0.0
            continue
        
        # Only trade in moderate volatility regime (ATR rank between 30-70)
        if atr_rank_12h_6h[i] < 30 or atr_rank_12h_6h[i] > 70:
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 6h Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 6h Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation: current 12h volume > 1.3x 24-period average
            # Need to get current 12h volume - approximate using 6h volume scaled
            # More accurate: use the last completed 12h bar's volume
            vol_ratio = volume[i] / vol_ma_24_6h[i] if vol_ma_24_6h[i] > 0 else 0
            
            # Enter long: price closes above 6h Donchian high with volume confirmation
            if (close[i] > donchian_high[i] and 
                vol_ratio > 1.3):
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below 6h Donchian low with volume confirmation
            elif (close[i] < donchian_low[i] and 
                  vol_ratio > 1.3):
                position = -1
                signals[i] = -0.25
    
    return signals