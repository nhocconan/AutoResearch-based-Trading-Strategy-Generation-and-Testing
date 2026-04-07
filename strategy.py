#!/usr/bin/env python3
"""
6h_Donchian_20_12h_Trend_Volume_v1
Hypothesis: On 6-hour timeframe, use Donchian channel breakouts with 12-hour trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high with 12h EMA(30) trending up and volume > 1.5x 20-period average.
Short when price breaks below 20-period Donchian low with 12h EMA(30) trending down and volume > 1.5x 20-period average.
Exit when price returns to the Donchian midpoint.
Designed for 15-30 trades/year to minimize fee decay while capturing strong trends with higher timeframe validation.
Works in both bull/bear markets as Donchian channels adapt to volatility and 12h trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian_20_12h_Trend_Volume_v1"
timeframe = "6h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(30) for trend filter
    close_12h = df_12h['close'].values
    ema_30_12h = pd.Series(close_12h).ewm(span=30, adjust=False, min_periods=30).mean().values
    ema_30_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_30_12h)
    
    # Determine 12h trend direction (using EMA slope)
    trend_up = np.zeros(len(ema_30_12h_aligned), dtype=bool)
    trend_down = np.zeros(len(ema_30_12h_aligned), dtype=bool)
    for i in range(1, len(ema_30_12h_aligned)):
        if not np.isnan(ema_30_12h_aligned[i]) and not np.isnan(ema_30_12h_aligned[i-1]):
            trend_up[i] = ema_30_12h_aligned[i] > ema_30_12h_aligned[i-1]
            trend_down[i] = ema_30_12h_aligned[i] < ema_30_12h_aligned[i-1]
    
    # Calculate Donchian Channel (20-period) on 6h timeframe
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 30), n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_30_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and 12h trend alignment
            if vol_ok:
                # Long: price breaks above Donchian high with 12h uptrend
                if (close[i] > donchian_high[i] and close[i-1] <= donchian_high[i-1] and 
                    trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with 12h downtrend
                elif (close[i] < donchian_low[i] and close[i-1] >= donchian_low[i-1] and 
                      trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals