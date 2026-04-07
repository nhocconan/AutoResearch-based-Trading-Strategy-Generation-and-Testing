#3/20183
#!/usr/bin/env python3
"""
4h_donchian_20_12h_trend_volume_v3
Hypothesis: On 4-hour timeframe, use Donchian channel breakout (20-period) with 12-hour EMA trend filter and volume confirmation.
Breakout above upper Donchian with 12h EMA(50) trending up and volume > 1.5x 20-period average triggers long.
Breakout below lower Donchian with 12h EMA(50) trending down and volume > 1.5x 20-period average triggers short.
Exit when price re-enters Donchian channel or on opposite breakout.
Designed for 20-50 trades/year to minimize fee drag while capturing strong trends with multi-timeframe confirmation.
Works in bull markets via breakout momentum and bear markets via short breakdowns with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_12h_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Determine 12h trend direction (using EMA slope)
    trend_up = np.zeros(len(ema_50_12h_aligned), dtype=bool)
    trend_down = np.zeros(len(ema_50_12h_aligned), dtype=bool)
    for i in range(1, len(ema_50_12h_aligned)):
        if not np.isnan(ema_50_12h_aligned[i]) and not np.isnan(ema_50_12h_aligned[i-1]):
            trend_up[i] = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            trend_down[i] = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
    
    # Calculate Donchian Channel (20-period) on 4h timeframe
    period = 20
    # Highest high over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    # Lowest low over period
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel (below upper) OR opposite breakout with trend
            if close[i] < highest_high[i]:
                position = 0
                signals[i] = 0.0
            elif (close[i] < lowest_low[i] and trend_down[i] and vol_ok):
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel (above low) OR opposite breakout with trend
            if close[i] > lowest_low[i]:
                position = 0
                signals[i] = 0.0
            elif (close[i] > highest_high[i] and trend_up[i] and vol_ok):
                position = 1
                signals[i] = 0.30
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Only enter with volume confirmation and 12h trend alignment
            if vol_ok:
                # Long: breakout above upper Donchian with 12h uptrend
                if (close[i] > highest_high[i] and close[i-1] <= highest_high[i-1] and 
                    trend_up[i]):
                    position = 1
                    signals[i] = 0.30
                # Short: breakout below lower Donchian with 12h downtrend
                elif (close[i] < lowest_low[i] and close[i-1] >= lowest_low[i-1] and 
                      trend_down[i]):
                    position = -1
                    signals[i] = -0.30
    
    return signals