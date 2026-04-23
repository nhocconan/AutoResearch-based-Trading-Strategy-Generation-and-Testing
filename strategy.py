#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Long when price breaks above Donchian upper band (20-period high) AND close > 1w EMA50 AND volume > 1.5x 20-period average
- Short when price breaks below Donchian lower band (20-period low) AND close < 1w EMA50 AND volume > 1.5x 20-period average
- Exit when price crosses Donchian midpoint (mean reversion to center)
- Uses 1w EMA50 for HTF trend alignment to avoid counter-trend entries in bear markets
- Volume confirmation threshold set to 1.5x to reduce false breakouts while maintaining sufficient trades
- Designed for both bull and bear markets: trend filter prevents counter-trend entries
- Target: 7-25 trades/year (30-100 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels from daily data (20-period)
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # We need to calculate Donchian on 1d data, but we're on 1d timeframe
    # So we can use the prices dataframe directly for Donchian calculation
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper[i]  # Break above upper band
        breakout_down = close[i] < donchian_lower[i]  # Break below lower band
        
        # Trend filter (using 1w EMA50)
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Donchian breakout up + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Donchian midpoint (mean reversion)
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint
                if close[i] < donchian_mid[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Price crosses above midpoint
                if close[i] > donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0