#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Choppiness Index regime filter + 4h Donchian breakout + 1d EMA34 trend
    # Choppiness Index identifies ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets
    # In trending regimes: Donchian breakout captures momentum
    # In ranging regimes: fade Donchian breakouts (mean reversion)
    # EMA34 on 1d filters for long-term trend direction to avoid counter-trend trades
    # Works in both bull and bear markets by adapting to regime
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian Channel (20-period) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Choppiness Index (14-period) on 4h
    # CHOP = 100 * log10(sum(ATR14) / (n * (max(high) - min(low)))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_14 = max_high_14 - min_low_14
    
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(sum_atr14 / (14 * range_14)) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((range_14 > 0) & (sum_atr14 > 0), chop_raw, 50.0)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            is_trending = chop[i] < 38.2
            is_ranging = chop[i] > 61.8
            
            if is_trending:
                # Trending regime: follow Donchian breakout with trend filter
                if close[i] > donchian_high[i] and close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and close[i] < ema34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # Ranging regime: fade Donchian breakouts (mean reversion)
                if close[i] > donchian_high[i] and close[i] < ema34_1d_aligned[i]:
                    # Price above Donchian high but below EMA34 (overbought in downtrend)
                    signals[i] = -0.25
                    position = -1
                elif close[i] < donchian_low[i] and close[i] > ema34_1d_aligned[i]:
                    # Price below Donchian low but above EMA34 (oversold in uptrend)
                    signals[i] = 0.25
                    position = 1
        else:
            # Exit conditions
            if position == 1:
                # Long exit: price returns to Donchian mid or trend reversal
                if close[i] < donchian_mid[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Short exit: price returns to Donchian mid or trend reversal
                if close[i] > donchian_mid[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Choppiness_Regime_Donchian_Breakout_1dEMA34_Trend_v1"
timeframe = "4h"
leverage = 1.0