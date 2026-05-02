#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h Supertrend combination with volume confirmation
# Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear conviction
# 12h Supertrend (ATR=10, mult=3.0) provides robust trend filter across market regimes
# Volume confirmation (>1.3x 50-period EMA) ensures institutional participation
# Works in bull markets (Supertrend up + Bull Power > 0) and bear markets (Supertrend down + Bear Power < 0)
# Targets 12-37 trades/year (50-150 total over 4 years) on 6h timeframe
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining return potential

name = "6h_ElderRay_12hSupertrend_Trend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for Supertrend trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h Supertrend (ATR=10, mult=3.0)
    # True Range calculation
    tr1 = pd.Series(df_12h['high']).values - pd.Series(df_12h['low']).values
    tr2 = np.abs(pd.Series(df_12h['high']).values - pd.Series(df_12h['close']).shift(1).values)
    tr3 = np.abs(pd.Series(df_12h['low']).values - pd.Series(df_12h['close']).shift(1).values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (pd.Series(df_12h['high']).values + pd.Series(df_12h['low']).values) / 2
    upperband = hl2 + (3.0 * atr)
    lowerband = hl2 - (3.0 * atr)
    
    supertrend = np.full_like(hl2, np.nan, dtype=np.float64)
    direction = np.full_like(hl2, np.nan, dtype=np.float64)  # 1 for uptrend, -1 for downtrend
    
    # Initialize
    supertrend[0] = upperband[0]
    direction[0] = 1
    
    for i in range(1, len(hl2)):
        if close[i-1] > supertrend[i-1]:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
        
        # Determine direction
        if close[i] > supertrend[i]:
            direction[i] = 1
        else:
            direction[i] = -1
    
    supertrend_12h_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Elder Ray (6h timeframe)
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation
    vol_ema_50 = pd.Series(volume).ewm(span=50, adjust=False, min_periods=50).mean().values
    volume_confirmation = volume > (1.3 * vol_ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend_12h_aligned[i]) or np.isnan(direction_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: 12h Supertrend uptrend + Bull Power positive + volume confirmation
            if (direction_12h_aligned[i] == 1) and (bull_power[i] > 0) and volume_confirmation[i]:
                signals[i] = 0.25
                position = 1
            # Short: 12h Supertrend downtrend + Bear Power negative + volume confirmation
            elif (direction_12h_aligned[i] == -1) and (bear_power[i] < 0) and volume_confirmation[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: 12h Supertrend turns down OR Bull Power turns negative
            if (direction_12h_aligned[i] == -1) or (bull_power[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: 12h Supertrend turns up OR Bear Power turns positive
            if (direction_12h_aligned[i] == 1) or (bear_power[i] >= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals