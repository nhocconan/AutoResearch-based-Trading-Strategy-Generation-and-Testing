#!/usr/bin/env python3
"""
6h_trix_volume_regime_v1
Hypothesis: TRIX (12-period) crossover with volume confirmation and regime filter (Choppiness Index < 61.8 = trending).
TRIX filters out insignificant price movements and is effective in trending markets.
Volume confirms breakout strength. Choppiness Index ensures we only trade in trending regimes,
avoiding whipsaws in ranging markets. Works in bull and bear by adapting to trend strength.
Target: 15-35 trades/year on 6h with strict entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for regime filter (Choppiness Index)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate TRIX on 6h close
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1 period ago
    ema1 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema2 = ema1.ewm(span=12, adjust=False).mean()
    ema3 = ema2.ewm(span=12, adjust=False).mean()
    trix = ema3.pct_change() * 100  # Percentage change
    
    # Calculate Choppiness Index on daily data
    # CHOP = 100 * log10(sum(ATR, n) / (max(high, n) - min(low, n))) / log10(n)
    atr_list = []
    for i in range(len(df_1d)):
        if i == 0:
            tr = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            tr = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
        atr_list.append(tr)
    
    atr = pd.Series(atr_list)
    chop_period = 14
    sum_atr = atr.rolling(window=chop_period, min_periods=chop_period).sum()
    max_high = df_1d['high'].rolling(window=chop_period, min_periods=chop_period).max()
    min_low = df_1d['low'].rolling(window=chop_period, min_periods=chop_period).min()
    
    chop = 100 * (np.log10(sum_atr) - np.log10(max_high - min_low)) / np.log10(chop_period)
    chop_values = chop.values
    
    # Align TRIX and Chop to 6h timeframe
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix.values)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_values)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(trix_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        vol_spike = volume[i] > (vol_ma[i] * 1.5)
        
        # Regime filter: Choppiness Index < 61.8 = trending market
        trending_regime = chop_aligned[i] < 61.8
        
        # TRIX signal: zero line cross
        trix_now = trix_aligned[i]
        trix_prev = trix_aligned[i-1] if i > 0 else 0
        
        bullish_cross = trix_prev <= 0 and trix_now > 0
        bearish_cross = trix_prev >= 0 and trix_now < 0
        
        if position == 1:  # Long position
            # Exit: TRIX turns bearish with volume or regime changes to ranging
            if bearish_cross and vol_spike or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX turns bullish with volume or regime changes to ranging
            if bullish_cross and vol_spike or not trending_regime:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: TRIX bullish cross + volume + trending regime
            if bullish_cross and vol_spike and trending_regime:
                position = 1
                signals[i] = 0.25
            # Enter short: TRIX bearish cross + volume + trending regime
            elif bearish_cross and vol_spike and trending_regime:
                position = -1
                signals[i] = -0.25
    
    return signals