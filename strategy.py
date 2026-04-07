#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v3
Hypothesis: On daily timeframe, use Donchian(20) breakouts with trend filter from weekly EMA200 and volume confirmation. Enter long on upper band breakout in uptrend with volume > 1.5x average, short on lower band breakdown in downtrend with volume > 1.5x average. Exit on opposite band touch. Weekly trend filter provides stronger regime filter than daily, reducing whipsaws. Designed for low frequency (7-25 trades/year) to avoid fee drag while capturing trend continuation in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v3"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    w_close = df_1w['close'].values
    w_ema200 = pd.Series(w_close).ewm(span=200, adjust=False).mean().values
    w_ema200_aligned = align_htf_to_ltf(prices, df_1w, w_ema200)
    
    # Calculate 20-period average volume for confirmation (daily)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if weekly EMA200 not available
        if np.isnan(w_ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs weekly EMA200
        uptrend = close[i] > w_ema200_aligned[i]
        downtrend = close[i] < w_ema200_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below lower Donchian(20)
            # Calculate Donchian lower band for last 20 periods
            if i >= 20:
                donchian_low = np.min(low[i-20:i])
                if close[i] <= donchian_low:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above upper Donchian(20)
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                if close[i] >= donchian_high:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Need at least 20 periods for Donchian calculation
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                donchian_low = np.min(low[i-20:i])
                
                # Long entry: price breaks above upper Donchian(20) in uptrend with volume confirmation
                long_entry = (close[i] > donchian_high) and uptrend and vol_confirm
                # Short entry: price breaks below lower Donchian(20) in downtrend with volume confirmation
                short_entry = (close[i] < donchian_low) and downtrend and vol_confirm
                
                if long_entry:
                    position = 1
                    signals[i] = 0.25
                elif short_entry:
                    position = -1
                    signals[i] = -0.25
    
    return signals