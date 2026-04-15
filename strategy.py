#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ehlers Fisher Transform on 1d HTF with volume confirmation
# Fisher Transform identifies extreme price movements and turning points.
# On 1d timeframe, it captures swing extremes that often reverse on 6h.
# Volume confirmation ensures breakouts have conviction.
# Works in both bull/bear markets as it identifies overextended moves.
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Ehlers Fisher Transform on 1d close prices
    # Formula: Fisher = 0.5 * ln((1+PriceNorm)/(1-PriceNorm)) where PriceNorm = (Price - Min)/(Max - Min) * 2 - 1
    hlc = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    max_hhc = pd.Series(hlc).rolling(window=10, min_periods=10).max().values
    min_hlc = pd.Series(hlc).rolling(window=10, min_periods=10).min().values
    range_hlc = max_hhc - min_hlc
    price_norm = np.where(range_hlc > 0, 2 * (hlc - min_hlc) / range_hlc - 1, 0)
    # Clamp to avoid division by zero in logit
    price_norm = np.clip(price_norm, -0.999, 0.999)
    fisher = 0.5 * np.log((1 + price_norm) / (1 - price_norm))
    # Smooth with 3-period EMA
    fisher_smooth = pd.Series(fisher).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Align Fisher Transform to 6h
    fisher_6h = align_htf_to_ltf(prices, df_1d, fisher_smooth)
    
    # Get 1d ATR for volatility regime filter
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = np.abs(df_1d['high'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr3 = np.abs(df_1d['low'] - np.concatenate([[df_1d['close'].iloc[0]], df_1d['close'].iloc[:-1]]))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_pct = atr_1d / df_1d['close'].values
    atr_regime = align_htf_to_ltf(prices, df_1d, atr_1d_pct)  # ATR as % of price
    
    # 6h volume confirmation (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    # Session filter: avoid low-volume Asian session (00-06 UTC) for better quality
    hours = prices.index.hour
    in_session = (hours >= 6) & (hours <= 23)  # 06:00-23:00 UTC
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(fisher_6h[i]) or np.isnan(atr_regime[i]) or 
            np.isnan(volume_ratio[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Fisher Transform crosses below -1.5 (oversold) AND
        # 2. ATR regime > 0.02 (sufficient volatility) AND
        # 3. Volume confirmation: volume > 1.5x average
        if (fisher_6h[i] < -1.5 and
            atr_regime[i] > 0.02 and
            volume_ratio[i] > 1.5):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Fisher Transform crosses above +1.5 (overbought) AND
        # 2. ATR regime > 0.02 (sufficient volatility) AND
        # 3. Volume confirmation: volume > 1.5x average
        elif (fisher_6h[i] > 1.5 and
              atr_regime[i] > 0.02 and
              volume_ratio[i] > 1.5):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_FisherTransform_Volume_Regime_v1"
timeframe = "6h"
leverage = 1.0