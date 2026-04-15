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
    
    # Calculate 1d ATR(14) for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 1d Bollinger Bands (20, 2) for mean reversion signals
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + 2.0 * std_20
    lower_band = sma_20 - 2.0 * std_20
    
    # Align Bollinger Bands to 12h
    upper_band_12h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_12h = align_htf_to_ltf(prices, df_1d, lower_band)
    sma_20_12h = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # Calculate 12h ATR(14) for volatility confirmation
    tr1_12h = high - low
    tr2_12h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_12h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    atr_14_12h = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 12h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(upper_band_12h[i]) or 
            np.isnan(lower_band_12h[i]) or np.isnan(sma_20_12h[i]) or 
            np.isnan(atr_14_12h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when daily ATR is elevated (> 0.8% of price)
        vol_regime = atr_14_1d_aligned[i] > 0.008 * close[i]
        
        # Long conditions:
        # 1. Price touches or breaks below lower Bollinger Band (oversold)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. 12h ATR > 0.5% of price (ensure sufficient volatility for bounce)
        # 4. Daily volatility regime filter (avoid low-volatility chop)
        if (close[i] <= lower_band_12h[i] and
            volume_ratio[i] > 1.5 and
            atr_14_12h[i] > 0.005 * close[i] and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price touches or breaks above upper Bollinger Band (overbought)
        # 2. Volume confirmation: volume > 1.5x average
        # 3. 12h ATR > 0.5% of price
        # 4. Daily volatility regime filter
        elif (close[i] >= upper_band_12h[i] and
              volume_ratio[i] > 1.5 and
              atr_14_12h[i] > 0.005 * close[i] and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Bollinger_Bands_Volume_Regime_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0