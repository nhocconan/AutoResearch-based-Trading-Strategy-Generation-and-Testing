#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1D Donchian Breakout with Weekly Trend and Volume Spike
# - Uses Donchian channels from weekly timeframe (upper/lower bands)
# - Breakout above weekly upper band with 1d uptrend or below weekly lower band with 1d downtrend
# - Volume spike confirms breakout strength
# - Works in bull/bear by using 1d trend filter to avoid counter-trend trades
# - Target: 10-25 trades/year to minimize fee drag on 1d timeframe
# - Focus on BTC/ETH as primary targets, avoids SOL-only bias

name = "1D_DonchianBreakout_1dTrend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Donchian calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels (20-period) using weekly data
    # Upper = max(high, 20), Lower = min(low, 20)
    n1w = len(high_1w)
    donchian_upper = np.full(n1w, np.nan)
    donchian_lower = np.full(n1w, np.nan)
    
    for i in range(20, n1w):
        donchian_upper[i] = np.max(high_1w[i-20:i])
        donchian_lower[i] = np.min(low_1w[i-20:i])
    
    # Align Donchian channels to 1d timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower)
    
    # 1d trend filter: EMA50 slope
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly upper band with 1d uptrend + volume spike
            long_cond = (close[i] > donchian_upper_aligned[i] and 
                        ema_50[i] > ema_50[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below weekly lower band with 1d downtrend + volume spike
            short_cond = (close[i] < donchian_lower_aligned[i] and 
                         ema_50[i] < ema_50[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly lower band
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly upper band
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals