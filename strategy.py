#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h 34-period EMA with 12h volatility regime filter and volume confirmation.
# Long when price > EMA34 AND 12h Bollinger Bands width > 60th percentile AND volume > 1.3x 6sma volume
# Short when price < EMA34 AND 12h Bollinger Bands width > 60th percentile AND volume > 1.3x 6sma volume
# Exit when price crosses back below/above EMA34
# Uses EMA34 for trend, BBW regime to avoid chop, volume for conviction. Works in trends and avoids false breakouts in low volatility.
# Target: 12-25 trades/year per symbol.
name = "6h_EMA34_VolRegime_Volume"
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
    
    # Get 12h data for Bollinger Band width regime filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Bollinger Bands (20, 2)
    close_12h = pd.Series(df_12h['close'])
    bb_mid = close_12h.rolling(window=20, min_periods=20).mean().values
    bb_std = close_12h.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Percentile rank of BB width (60th percentile threshold)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    bb_width_regime = bb_width_percentile > 0.6  # True when volatility is high (top 40%)
    
    # Align BB width regime to 6h
    bb_width_regime_aligned = align_htf_to_ltf(prices, df_12h, bb_width_regime.astype(float))
    
    # EMA34 on 6h close
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: volume > 1.3x 6-period SMA volume
    vol_ma6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_confirmed = volume > 1.3 * vol_ma6
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 50, 6)  # EMA34, BB, percentile, vol ma
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34[i]) or np.isnan(bb_width_regime_aligned[i]) or 
            np.isnan(vol_ma6[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema34[i]
        vol_regime = bb_width_regime_aligned[i] > 0.5  # boolean from aligned
        vol_conf = volume_confirmed[i]
        vol = volume[i]
        vol_ma = vol_ma6[i]
        
        if position == 0:
            # Long entry: price > EMA34 AND high volatility regime AND volume confirmation
            if price > ema and vol_regime and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short entry: price < EMA34 AND high volatility regime AND volume confirmation
            elif price < ema and vol_regime and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below EMA34
            if price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above EMA34
            if price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals