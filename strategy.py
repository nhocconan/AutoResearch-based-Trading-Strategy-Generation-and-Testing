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
    
    # Get weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian(20) channels for trend filter
    donchian_high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily
    dh_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    dl_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Calculate daily ATR(14) for volatility regime filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(dh_20_aligned[i]) or np.isnan(dl_20_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when ATR > 1.5% of price
        # This focuses on momentum days and avoids low-volatility chop
        vol_regime = atr_14[i] > 0.015 * close[i]
        
        # Long conditions:
        # 1. Price above weekly Donchian high (bullish breakout)
        # 2. Volume confirmation: volume > 2.0x average
        # 3. Volatility regime filter
        if (close[i] > dh_20_aligned[i] and
            volume_ratio[i] > 2.0 and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price below weekly Donchian low (bearish breakout)
        # 2. Volume confirmation: volume > 2.0x average
        # 3. Volatility regime filter
        elif (close[i] < dl_20_aligned[i] and
              volume_ratio[i] > 2.0 and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Weekly_Donchian20_Breakout_Volume_Regime_v2"
timeframe = "1d"
leverage = 1.0