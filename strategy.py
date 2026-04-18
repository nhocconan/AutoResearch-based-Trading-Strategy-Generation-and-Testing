#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation
Hypothesis: Weekly Donchian channel breakouts on the daily chart capture strong
trend moves that persist across market regimes. Volume confirmation filters
false breakouts, and we use the weekly trend (EMA34) to ensure we trade with
the dominant higher timeframe momentum. This strategy targets 15-25 trades
per year to minimize fee drag while capturing major moves in BTC, ETH, and SOL.
"""

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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Weekly EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.8x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        trend = ema34_1w_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume, in weekly uptrend
            if price > donchian_high[i] and vol_filter[i] and price > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume, in weekly downtrend
            elif price < donchian_low[i] and vol_filter[i] and price < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to Donchian mid-point or trend weakens
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if price < donchian_mid or price < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to Donchian mid-point or trend weakens
            donchian_mid = (donchian_high[i] + donchian_low[i]) / 2.0
            if price > donchian_mid or price > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0