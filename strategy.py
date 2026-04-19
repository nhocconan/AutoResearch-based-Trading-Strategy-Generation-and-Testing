#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
# Long when: price closes above upper Donchian, volume > 1.5x 20-bar avg, 12h EMA34 > 12h EMA34 of 10 bars ago
# Short when: price closes below lower Donchian, volume > 1.5x 20-bar avg, 12h EMA34 < 12h EMA34 of 10 bars ago
# Exit when price returns to 20-bar SMA or reverses to opposite Donchian band.
# Trend filter ensures trades only in trending markets, reducing whipsaw in ranging conditions.
# Target: 20-30 trades/year per symbol to minimize fee drag while capturing strong trends.
name = "4h_Donchian20_Volume_EMA34Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h data
    ema34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 4h timeframe
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Donchian channels (20-period)
    # Upper band: highest high of last 20 periods
    # Lower band: lowest low of last 20 periods
    # Middle: 20-period SMA
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    upper_band = high_series.rolling(window=20, min_periods=20).max().values
    lower_band = low_series.rolling(window=20, min_periods=20).min().values
    middle_band = close_series.rolling(window=20, min_periods=20).mean().values
    
    # Volume average (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_12h_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Trend condition: EMA34 rising (bullish) or falling (bearish)
        # Compare current EMA34 to EMA34 from 10 periods ago (equivalent to ~5 days on 12h)
        trend_bullish = ema34_12h_aligned[i] > ema34_12h_aligned[i-10] if i >= 10 else False
        trend_bearish = ema34_12h_aligned[i] < ema34_12h_aligned[i-10] if i >= 10 else False
        
        if position == 0:
            # Long breakout: price closes above upper Donchian with volume confirmation and bullish trend
            if price > upper_band[i] and vol > 1.5 * vol_ma and trend_bullish:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below lower Donchian with volume confirmation and bearish trend
            elif price < lower_band[i] and vol > 1.5 * vol_ma and trend_bearish:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to middle band or breaks below lower band
            if price <= middle_band[i] or price < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle band or breaks above upper band
            if price >= middle_band[i] or price > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals