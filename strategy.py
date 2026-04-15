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
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for regime filter
    delta = df_1d['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_14_1d = (100 - (100 / (1 + rs))).fillna(50).values
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 6h Donchian(20) breakout levels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi_14_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: RSI between 40-60 (avoid extremes)
        regime_filter = (rsi_14_1d_aligned[i] >= 40) and (rsi_14_1d_aligned[i] <= 60)
        
        # Volatility filter: ATR > 0.5% of price
        vol_filter = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        # Volume filter: volume > 1.2x average
        vol_ratio_filter = volume_ratio[i] > 1.2
        
        # Long conditions:
        # 1. Price breaks above 6h Donchian(20) high
        # 2. Volume confirmation
        # 3. Regime filter (not overbought/oversold)
        # 4. Volatility filter
        if (close[i] > highest_20[i] and
            vol_ratio_filter and
            regime_filter and
            vol_filter):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 6h Donchian(20) low
        # 2. Volume confirmation
        # 3. Regime filter (not overbought/oversold)
        # 4. Volatility filter
        elif (close[i] < lowest_20[i] and
              vol_ratio_filter and
              regime_filter and
              vol_filter):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Donchian20_Breakout_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0