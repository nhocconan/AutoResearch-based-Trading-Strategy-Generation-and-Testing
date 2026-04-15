#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly EMA34 trend filter and ATR-based position sizing
# Works in bull/bear: breakouts capture trends, EMA34 filter avoids counter-trend trades, ATR sizing adapts to volatility
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility-based position sizing
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d Donchian(20) channels
    donchian_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Position size inversely proportional to volatility (ATR), capped at 0.30
        vol_norm = atr_14_1d_aligned[i] / close[i]
        base_size = 0.30
        size = base_size * (0.01 / vol_norm)  # Scale to ~1% daily vol
        size = min(max(size, 0.10), 0.30)     # Clamp between 0.10 and 0.30
        
        # Long: price breaks above Donchian high AND weekly EMA34 uptrend (price > EMA)
        if (close[i] > donchian_high_20_aligned[i] and 
            close[i] > ema_34_1w_aligned[i]):
            signals[i] = size
            
        # Short: price breaks below Donchian low AND weekly EMA34 downtrend (price < EMA)
        elif (close[i] < donchian_low_20_aligned[i] and 
              close[i] < ema_34_1w_aligned[i]):
            signals[i] = -size
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_WeeklyEMA34_VolSized_v1"
timeframe = "1d"
leverage = 1.0