#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Donchian breakout with volume confirmation and ATR-based volatility filter.
# In bull markets: price breaks above weekly high with volume -> long.
# In bear markets: price breaks below weekly low with volume -> short.
# Volatility filter avoids choppy low-ATR periods. Weekly timeframe reduces trade frequency.
# Discrete position size 0.25 to limit drawdown and fees.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    highest_20 = df_1w['high'].rolling(window=20, min_periods=20).max().values
    lowest_20 = df_1w['low'].rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe (completed weekly bars only)
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Weekly ATR(14) for volatility regime filter
    tr1 = df_1w['high'] - df_1w['low']
    tr2 = np.abs(df_1w['high'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr3 = np.abs(df_1w['low'] - np.concatenate([[df_1w['close'].iloc[0]], df_1w['close'].iloc[:-1]]))
    tr_1w = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1w = pd.Series(tr_1w).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Daily volume ratio (current vs 20-day average) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: trade only when weekly ATR is elevated (> 0.8% of price)
        vol_regime = atr_14_1w_aligned[i] > 0.008 * close[i]
        
        # Long: price breaks above weekly Donchian high with volume confirmation
        if (close[i] > highest_20_aligned[i] and 
            volume_ratio[i] > 1.8 and 
            vol_regime):
            signals[i] = 0.25
            
        # Short: price breaks below weekly Donchian low with volume confirmation
        elif (close[i] < lowest_20_aligned[i] and 
              volume_ratio[i] > 1.8 and 
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Volume_VolatilityFilter"
timeframe = "1d"
leverage = 1.0