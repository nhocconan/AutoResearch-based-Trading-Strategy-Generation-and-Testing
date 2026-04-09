#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with weekly trend filter (price > EMA50 weekly) and volume confirmation
# Uses Donchian breakouts on daily timeframe for entry signals in the direction of weekly EMA50 trend
# Volume confirmation ensures breakouts have participation
# Weekly EMA50 adapts to longer-term trend, reducing whipsaw in ranging markets
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while maintaining edge
# Works in both bull/bear: weekly EMA50 captures intermediate trend, Donchian catches momentum bursts

name = "1d_1w_donchian_ema_volume_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for EMA50 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 with proper min_periods
    close_1w = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period Donchian channels on daily
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR trend turns bearish (price < EMA50 weekly)
            if close[i] < donchian_low[i] or close[i] < ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR trend turns bullish (price > EMA50 weekly)
            if close[i] > donchian_high[i] or close[i] > ema_50_1d[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout in direction of EMA50 trend with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian high AND price > EMA50 (bullish trend)
                if close[i] > donchian_high[i] and close[i] > ema_50_1d[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low AND price < EMA50 (bearish trend)
                elif close[i] < donchian_low[i] and close[i] < ema_50_1d[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals