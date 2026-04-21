#!/usr/bin/env python3
"""
6h_HTF_DailyVolatilityRegime_DonchianBreakout
Hypothesis: Use 1d ATR ratio (ATR(5)/ATR(20)) as volatility regime filter + 6h Donchian(20) breakout with volume confirmation.
In high volatility regimes (ratio > 1.2), trade breakouts; in low volatility (ratio < 0.8), fade reversals at Donchian middle.
Works in bull (breakouts capture momentum) and bear (mean reversion in low vol captures bounces) via regime adaptation.
Target: 12-25 trades/year per symbol. Uses discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Volatility Regime: ATR(5)/ATR(20) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_5 = pd.Series(tr).rolling(window=5, min_periods=5).mean().values
    atr_20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    vol_ratio = atr_5 / atr_20  # >1.2 = high vol, <0.8 = low vol
    
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # === 6h Donchian Channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian bands
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_high + donchian_low) / 2.0
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        vr = vol_ratio_aligned[i]
        
        if position == 0:
            # High volatility regime: trade breakouts
            if vr > 1.2:
                if price > donchian_high[i-1] and vol_ok:
                    signals[i] = 0.25
                    position = 1
                elif price < donchian_low[i-1] and vol_ok:
                    signals[i] = -0.25
                    position = -1
            # Low volatility regime: mean reversion at middle
            elif vr < 0.8:
                if price < donchian_middle[i-1] and price > donchian_low[i-1]:
                    # Long from lower half toward middle
                    signals[i] = 0.25
                    position = 1
                elif price > donchian_middle[i-1] and price < donchian_high[i-1]:
                    # Short from upper half toward middle
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions
            if vr > 1.2:
                # High vol: exit on Donchian middle break or opposite signal
                if price < donchian_middle[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Low vol: exit when price reaches opposite band or middle
                if price >= donchian_high[i-1] or price <= donchian_middle[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions
            if vr > 1.2:
                # High vol: exit on Donchian middle break or opposite signal
                if price > donchian_middle[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Low vol: exit when price reaches opposite band or middle
                if price <= donchian_low[i-1] or price >= donchian_middle[i-1]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_HTF_DailyVolatilityRegime_DonchianBreakout"
timeframe = "6h"
leverage = 1.0