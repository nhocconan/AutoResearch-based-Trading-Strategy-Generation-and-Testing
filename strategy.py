#!/usr/bin/env python3
"""
4h_donchian_breakout_12h_trend_volume_v2
Hypothesis: On 4-hour timeframe, buy when price breaks above 20-period Donchian upper band with 12h EMA(21) uptrend and volume above average; sell when price breaks below 20-period Donchian lower band with 12h EMA(21) downtrend and volume above average. Uses volume confirmation to avoid false breakouts and trend filter to align with higher timeframe momentum. Designed for 75-200 total trades over 4 years (~19-50/year) to minimize fee drag while capturing trend moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_12h_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter (20-period average)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish breakout: price above Donchian upper band with 12h uptrend
                if close[i] > donchian_high[i] and ema_12h_aligned[i] > close_12h[-1] if len(close_12h) > 0 else False:
                    # Simplified: use EMA > previous EMA for uptrend
                    if i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] > ema_12h_aligned[i-1]:
                        position = 1
                        signals[i] = 0.25
                # Bearish breakout: price below Donchian lower band with 12h downtrend
                elif close[i] < donchian_low[i] and ema_12h_aligned[i] < close_12h[-1] if len(close_12h) > 0 else False:
                    # Simplified: use EMA < previous EMA for downtrend
                    if i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] < ema_12h_aligned[i-1]:
                        position = -1
                        signals[i] = -0.25
    
    return signals