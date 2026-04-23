#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation.
- Long: price breaks above 20-period Donchian high + volume > 1.5x 20-period avg + ATR(14) < 0.03 * price (low volatility regime)
- Short: price breaks below 20-period Donchian low + volume > 1.5x 20-period avg + ATR(14) < 0.03 * price (low volatility regime)
- Exit: price re-enters Donchian channel (mean reversion) OR ATR spikes above 0.05 * price (high volatility)
- Uses Donchian for structure, volume for conviction, ATR for regime filter (low vol breakouts)
- Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in low vol uptrend) and bear (sell breakdowns in low vol downtrend)
- Avoids high volatility periods where breakouts fail
"""

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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Donchian channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d ATR for HTF regime filter (optional: can use same ATR if preferred)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14)  # Need 20 for Donchian, 14 for ATR
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(atr[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price and ATR as fraction of price for regime filter
        atr_frac = atr[i] / close[i] if close[i] > 0 else 1.0
        atr_1d_frac = atr_1d_aligned[i] / close[i] if close[i] > 0 else 1.0
        
        # Low volatility regime: ATR < 3% of price (calm market)
        low_vol = atr_frac < 0.03
        # High volatility regime: ATR > 5% of price (panic/exhaustion)
        high_vol = atr_frac > 0.05
        # HTF low vol: 1d ATR < 4% of price
        htf_low_vol = atr_1d_frac < 0.04
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + low vol + HTF low vol
            if (close[i] > donch_high[i] and 
                volume_confirm and 
                low_vol and 
                htf_low_vol):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume + low vol + HTF low vol
            elif (close[i] < donch_low[i] and 
                  volume_confirm and 
                  low_vol and 
                  htf_low_vol):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below Donchian low (mean reversion) OR high volatility spike
            if close[i] < donch_low[i] or high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above Donchian high (mean reversion) OR high volatility spike
            if close[i] > donch_high[i] or high_vol:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Volume_ATRRegime_LowVol"
timeframe = "12h"
leverage = 1.0