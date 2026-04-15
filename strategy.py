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
    
    # Calculate 1d Bollinger Bands (20, 2.0) for regime detection
    sma_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # Calculate 1d Donchian Channel (20) for breakout signals
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 6h ATR(14) for stoploss and volatility filter
    tr1_6h = high - low
    tr2_6h = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3_6h = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    atr_14_6h = pd.Series(tr_6h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_14_1d_aligned[i]) or np.isnan(bb_width_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_14_6h[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade when Bollinger Band width is in normal range
        # Avoids extremely low volatility (chop) and extremely high volatility (panic)
        vol_regime = (bb_width_aligned[i] > 0.01) & (bb_width_aligned[i] < 0.10)
        
        # Long conditions:
        # 1. Price breaks above 1d Donchian high with volume
        # 2. Volume confirmation: volume > 1.5x average
        # 3. 6h ATR > 0.3% of price (ensure sufficient volatility)
        # 4. Volatility regime filter (avoid chop and panic)
        if (close[i] > donchian_high_aligned[i] and
            volume_ratio[i] > 1.5 and
            atr_14_6h[i] > 0.003 * close[i] and
            vol_regime):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below 1d Donchian low with volume
        # 2. Volume confirmation: volume > 1.5x average
        # 3. 6h ATR > 0.3% of price
        # 4. Volatility regime filter
        elif (close[i] < donchian_low_aligned[i] and
              volume_ratio[i] > 1.5 and
              atr_14_6h[i] > 0.003 * close[i] and
              vol_regime):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_Volume_Regime_v1"
timeframe = "1d"
leverage = 1.0