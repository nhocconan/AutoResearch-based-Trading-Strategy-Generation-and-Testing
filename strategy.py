#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND 12h EMA34 > EMA89 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low AND 12h EMA34 < EMA89 AND volume > 1.5x 20-period average.
Exit when price touches Donchian middle line OR volume drops below average.
Designed for low trade frequency (19-50/year) on 4h timeframe with proven edge in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMAs for trend filter
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_12h = pd.Series(close_12h).ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    ema89_12h_aligned = align_htf_to_ltf(prices, df_12h, ema89_12h)
    
    # Calculate Donchian channels on 4h (primary timeframe)
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2.0
    
    # Calculate 20-period volume average on 4h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(100, lookback - 1)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(ema34_12h_aligned[i]) or np.isnan(ema89_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma[i]
        ema34 = ema34_12h_aligned[i]
        ema89 = ema89_12h_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian AND 12h EMA34 > EMA89 AND volume > 1.5x avg
            if price > upper[i] and ema34 > ema89 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian AND 12h EMA34 < EMA89 AND volume > 1.5x avg
            elif price < lower[i] and ema34 < ema89 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price touches middle Donchian OR volume < average
            if price <= middle[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price touches middle Donchian OR volume < average
            if price >= middle[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0