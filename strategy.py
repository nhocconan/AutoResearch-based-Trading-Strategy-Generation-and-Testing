# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe strategy using daily ATR-based volatility regime detection combined with 
price position relative to daily VWAP for mean reversion in high volatility and trend following 
in low volatility. Daily ATR percentile identifies regime: high vol (>80th percentile) triggers 
mean reversion at VWAP deviations, low vol (<20th percentile) triggers trend following with 
price action relative to daily open. Volume confirmation filters both regimes. 
Expected to work in bull/bear by adapting to volatility conditions rather than fixed direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 30):
        return np.zeros(n)
    
    # === Daily ATR for volatility regime ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile (252-day lookback for regime)
    atr_percentile = pd.Series(atr_14).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # Align ATR percentile to 6h timeframe
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # === Daily VWAP for mean reversion/trend reference ===
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = pd.Series(typical_price_1d * df_1d['volume'].values).cumsum().values
    vwap_denominator = pd.Series(df_1d['volume'].values).cumsum().values
    vwap_1d = vwap_numerator / vwap_denominator
    # Handle first value
    vwap_1d[0] = typical_price_1d[0]
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === Daily open for trend reference in low volatility ===
    open_1d = df_1d['open'].values
    open_1d_aligned = align_htf_to_ltf(prices, df_1d, open_1d)
    
    # === Volume confirmation (20-period average) ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(atr_percentile_aligned[i]) or 
            np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(open_1d_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        atr_percentile_val = atr_percentile_aligned[i]
        vwap_val = vwap_1d_aligned[i]
        open_val = open_1d_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # High volatility regime: mean reversion at VWAP deviations
            if atr_percentile_val > 80:  # High volatility
                if price_close < vwap_val * 0.998 and vol_ratio_val > 1.3:  # Below VWAP
                    signals[i] = 0.25
                    position = 1
                elif price_close > vwap_val * 1.002 and vol_ratio_val > 1.3:  # Above VWAP
                    signals[i] = -0.25
                    position = -1
            # Low volatility regime: trend following with daily open
            elif atr_percentile_val < 20:  # Low volatility
                if price_close > open_val and vol_ratio_val > 1.2:  # Above open
                    signals[i] = 0.25
                    position = 1
                elif price_close < open_val and vol_ratio_val > 1.2:  # Below open
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                # Long exit: price crosses above VWAP (mean reversion) or below open (trend fail)
                if (atr_percentile_val > 80 and price_close > vwap_val) or \
                   (atr_percentile_val < 20 and price_close < open_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price crosses below VWAP (mean reversion) or above open (trend fail)
                if (atr_percentile_val > 80 and price_close < vwap_val) or \
                   (atr_percentile_val < 20 and price_close > open_val):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ATR_Volatility_Regime_VWAP_Open"
timeframe = "6h"
leverage = 1.0