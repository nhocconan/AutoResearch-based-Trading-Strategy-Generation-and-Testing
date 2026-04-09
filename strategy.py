#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout + 4h volume confirmation + 1d trend filter
# Camarilla levels provide precise intraday support/resistance; breakouts with volume confirm institutional participation
# 1d EMA200 filters for higher timeframe trend alignment to avoid counter-trend whipsaws
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Target: 60-150 total trades over 4 years (15-37/year) with discrete sizing 0.20

name = "1h_4h_1d_camarilla_breakout_volume_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume for 4h
    vol_4h = df_4h['volume'].values
    avg_vol_4h = np.full(len(vol_4h), np.nan)
    for i in range(len(vol_4h)):
        if i < 20:
            avg_vol_4h[i] = np.nan
        else:
            avg_vol_4h[i] = np.mean(vol_4h[i-20:i])
    
    # Align 4h average volume to 1h (wait for 4h bar close)
    avg_vol_4h_aligned = align_htf_to_ltf(prices, df_4h, avg_vol_4h)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        multiplier = 2 / (200 + 1)
        ema200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200[i] = (close_1d[i] * multiplier) + (ema200[i-1] * (1 - multiplier))
    
    # Align 1d EMA200 to 1h (wait for daily close)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid or outside session
        if (np.isnan(avg_vol_4h_aligned[i]) or np.isnan(ema200_aligned[i]) or 
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for 1h using previous bar's range
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_val = prev_high - prev_low
            
            if range_val > 0:
                # Camarilla levels
                h5 = prev_close + (range_val * 1.1 / 2)
                h4 = prev_close + (range_val * 1.1 / 4)
                h3 = prev_close + (range_val * 1.1 / 6)
                l3 = prev_close - (range_val * 1.1 / 6)
                l4 = prev_close - (range_val * 1.1 / 4)
                l5 = prev_close - (range_val * 1.1 / 2)
                
                # Volume confirmation: current volume > 1.5x 4h average volume
                volume_confirmed = volume[i] > 1.5 * avg_vol_4h_aligned[i]
                
                if position == 1:  # Long position
                    # Exit: price < L3 (Camarilla support) OR trend turns bearish
                    if close[i] < l3 or close[i] < ema200_aligned[i]:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = 0.20
                        
                elif position == -1:  # Short position
                    # Exit: price > H3 (Camarilla resistance) OR trend turns bullish
                    if close[i] > h3 or close[i] > ema200_aligned[i]:
                        position = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -0.20
                else:  # Flat
                    # Entry logic with volume confirmation and Camarilla breakout + trend filter
                    if volume_confirmed:
                        # Long entry: price > H3 AND bullish trend (price > EMA200)
                        if close[i] > h3 and close[i] > ema200_aligned[i]:
                            position = 1
                            signals[i] = 0.20
                        # Short entry: price < L3 AND bearish trend (price < EMA200)
                        elif close[i] < l3 and close[i] < ema200_aligned[i]:
                            position = -1
                            signals[i] = -0.20
    
    return signals