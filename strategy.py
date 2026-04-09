#!/usr/bin/env python3
# 6h_keltner_donchian_breakout_v2
# Hypothesis: 6h Donchian breakout with Keltner channel filter and volume confirmation.
# Works in bull/bear: Donchian captures breakouts, Keltner filters false signals in low volatility,
# volume confirms institutional participation. Uses discrete sizing to minimize fees.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_keltner_donchian_breakout_v2"
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
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h Donchian Channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h Keltner Channel (20, ATR=10, multiplier=2.0)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    keltner_upper = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values + 2.0 * atr
    keltner_lower = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values - 2.0 * atr
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR below Keltner lower
            if close[i] < donchian_low[i] or close[i] < keltner_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR above Keltner upper
            if close[i] > donchian_high[i] or close[i] > keltner_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Need volume confirmation and trend alignment
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            price_above_ema = close[i] > ema50_1d_aligned[i]
            price_below_ema = close[i] < ema50_1d_aligned[i]
            
            if volume_confirmed:
                # Long: price breaks above Donchian high AND above Keltner upper AND above 1d EMA50
                if close[i] > donchian_high[i] and close[i] > keltner_upper[i] and price_above_ema:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low AND below Keltner lower AND below 1d EMA50
                elif close[i] < donchian_low[i] and close[i] < keltner_lower[i] and price_below_ema:
                    position = -1
                    signals[i] = -0.25
    
    return signals