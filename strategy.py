#!/usr/bin/env python3
"""
12h_PriceChannel_Squeeze_Breakout_With_1dTrend
Hypothesis: In both bull and bear markets, periods of low volatility (squeeze) precede explosive moves.
We identify squeeze using Bollinger Band Width at a 1-day lookback period. When BBW is at its
20-period low (indicating compression) and price breaks out of the Donchian channel (20-period)
on the 12-hour timeframe with volume confirmation, we enter in the direction of the breakout.
Trend filter uses 1-day EMA34 to avoid counter-trend moves. This captures momentum after
consolidation, which works in trending and mean-reverting regimes by only trading breakouts
of volatility contractions.
Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag while capturing
significant moves.
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
    
    # 1-day data for Bollinger Band Width and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Bollinger Bands on 1-day close (20, 2)
    close_1d = df_1d['close'].values
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Bollinger Band Width 20-period low (squeeze signal)
    bbw_low = pd.Series(bb_width).rolling(window=20, min_periods=20).min().values
    squeeze = bb_width <= bbw_low  # True when at or near 20-period low
    
    # 1-day EMA34 trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1-day indicators to 12h timeframe
    squeeze_12h = align_htf_to_ltf(prices, df_1d, squeeze)
    ema_34_12h = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 12-hour Donchian channel (20-period) for breakout
    # We calculate directly on 12h prices since we're already in that timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Warmup for 20-period indicators
    
    for i in range(start_idx, n):
        if (np.isnan(squeeze_12h[i]) or np.isnan(ema_34_12h[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        squeeze_active = squeeze_12h[i]
        ema_trend = ema_34_12h[i]
        upper_donchian = highest_20[i]
        lower_donchian = lowest_20[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: squeeze break above upper Donchian with volume in uptrend
            if squeeze_active and price > upper_donchian and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: squeeze break below lower Donchian with volume in downtrend
            elif squeeze_active and price < lower_donchian and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: hold until price returns to lower Donchian or trend breaks
            signals[i] = 0.25
            if price < lower_donchian or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short: hold until price returns to upper Donchian or trend breaks
            signals[i] = -0.25
            if price > upper_donchian or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_PriceChannel_Squeeze_Breakout_With_1dTrend"
timeframe = "12h"
leverage = 1.0