#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v2
Hypothesis: On 1-day timeframe, use Donchian channel breakout from weekly timeframe with volume confirmation and trend filter.
Long when price breaks above weekly Donchian(20) high with daily volume > 1.5x 20-day average and price above weekly EMA(50).
Short when price breaks below weekly Donchian(20) low with daily volume > 1.5x 20-day average and price below weekly EMA(50).
Exit when price returns to weekly Donchian midpoint or opposite band.
Designed for 10-25 trades/year to minimize fee fade while capturing strong trends.
Works in both bull/bear markets as weekly Donchian adapts to volatility and volume filter ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v2"
timeframe = "1d"
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
    
    # Get weekly data for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channel (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian bands
    upper_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    middle_20 = (upper_20 + lower_20) / 2
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all weekly data to daily timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, lower_20)
    middle_20_aligned = align_htf_to_ltf(prices, df_1w, middle_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: 20-period average on daily timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(middle_20_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to weekly Donchian midpoint or breaks below lower band
            if close[i] <= middle_20_aligned[i] or close[i] < lower_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly Donchian midpoint or breaks above upper band
            if close[i] >= middle_20_aligned[i] or close[i] > upper_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price breaks above weekly Donchian upper band with price above weekly EMA(50)
                if close[i] > upper_20_aligned[i] and close[i] > ema_50_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below weekly Donchian lower band with price below weekly EMA(50)
                elif close[i] < lower_20_aligned[i] and close[i] < ema_50_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals