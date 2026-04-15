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
    
    # Calculate weekly Donchian(20) for trend filter
    highest_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    
    # Calculate daily ATR(14) for volatility regime and position sizing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # True Range calculation
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: only long in uptrend, only short in downtrend
        weekly_uptrend = close[i] > donchian_high_aligned[i]
        weekly_downtrend = close[i] < donchian_low_aligned[i]
        
        # Volatility regime filter: trade when volatility is elevated
        vol_regime = atr_14_1d_aligned[i] > 0.008 * close[i]
        
        # Volume confirmation: significant volume spike
        vol_confirm = volume_ratio[i] > 2.0
        
        # Long conditions: weekly uptrend + volatility regime + volume confirmation
        if weekly_uptrend and vol_regime and vol_confirm:
            signals[i] = 0.25  # 25% position
            
        # Short conditions: weekly downtrend + volatility regime + volume confirmation
        elif weekly_downtrend and vol_regime and vol_confirm:
            signals[i] = -0.25  # 25% short
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian20_Volume_VolatilityFilter_v3"
timeframe = "1d"
leverage = 1.0