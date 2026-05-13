#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter, volume confirmation (>1.5x 20-bar avg volume), and choppiness regime filter (CHOP < 61.8 = trending).
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 12h timeframe.
# Donchian breakouts capture momentum; 1d EMA34 ensures higher timeframe trend alignment;
# Volume confirmation filters low-participation breakouts; Chop filter avoids whipsaws in ranging markets.
# Designed for fewer, higher-quality trades to minimize fee drag while working in both bull and bear markets.

name = "12h_Donchian20_1dEMA34_Volume_Chop_Filter_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) from prior candle only
    lookback_dc = 20
    prior_high_max = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().shift(1).values
    prior_low_min = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().shift(1).values
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    # Calculate Choppiness Index (14-period) for regime filter
    lookback_chop = 14
    tr1 = pd.Series(high).rolling(lookback_chop).max() - pd.Series(low).rolling(lookback_chop).min()
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=lookback_chop, min_periods=lookback_chop).sum()
    sum_close_diff = abs(pd.Series(close) - pd.Series(close).shift(1)).rolling(window=lookback_chop, min_periods=lookback_chop).sum()
    chop = 100 * np.log10(sum_close_diff / atr) / np.log10(lookback_chop)
    chop_values = chop.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_dc, lookback_vol, lookback_chop, 1), n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(prior_high_max[i]) or np.isnan(prior_low_min[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(chop_values[i])):
            signals[i] = 0.0
            continue
        
        # Choppiness regime filter: only trade in trending markets (CHOP < 61.8)
        if chop_values[i] >= 61.8:
            # In ranging market, force flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band, close > 1d EMA34, volume spike
            if (high[i] > prior_high_max[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band, close < 1d EMA34, volume spike
            elif (low[i] < prior_low_min[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower band OR volume drops below average
            if (low[i] < prior_low_min[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper band OR volume drops below average
            if (high[i] > prior_high_max[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals