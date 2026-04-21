#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime
Hypothesis: 12h Camarilla R1/S1 breakout with 1d HTF trend filter (price > EMA34 for long bias, < EMA34 for short bias) 
captures strong directional moves with low trade frequency. Volume confirmation (>1.5x 20-period average) filters weak breakouts. 
Choppiness regime filter (CHOP > 61.8 = range, < 38.2 = trend) ensures trades only in trending markets. 
ATR(14) trailing stop via signal=0 when price moves against position by 2.0*ATR. 
Designed for 12h timeframe to achieve 12-37 trades/year (50-150 total over 4 years) minimizing fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop (1d for EMA trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === Primary timeframe (12h) indicators ===
    # Use 12h close prices for calculations (prices DataFrame is already 12h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla (using previous bar's OHLC)
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan  # First value invalid
    
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    # Using previous bar's range
    prev_range = prev_high - prev_low
    R1 = prev_close + 1.1 * prev_range / 12
    S1 = prev_close - 1.1 * prev_range / 12
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_ma
    
    # ATR (14-period) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = np.nan
    tr2[0] = np.nan
    tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (CHOP) regime filter - using 14-period
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high,14) - min(low,14))) / log10(14)
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denom = max_high_14 - min_low_14
    # Avoid division by zero
    chop_denom = np.where(chop_denom == 0, 1e-10, chop_denom)
    chop = 100 * np.log10(atr_14 / chop_denom) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup period
        # Skip if indicators not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(volume_threshold[i]) 
            or np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        in_trending_regime = chop[i] < 38.2
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + long HTF bias + trending regime
            if price > R1[i] and volume[i] > volume_threshold[i] and price > ema_34_1d_aligned[i] and in_trending_regime:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + short HTF bias + trending regime
            elif price < S1[i] and volume[i] > volume_threshold[i] and price < ema_34_1d_aligned[i] and in_trending_regime:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below R1 (breakout failed)
            elif price < R1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above S1 (breakout failed)
            elif price > S1[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0