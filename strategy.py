#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly Donchian breakout with volume confirmation and weekly trend filter
# Works in bull markets via breakout momentum, in bear via short breakdowns
# Weekly trend filter avoids counter-trend trades, volume confirms institutional interest
# Target: 20-40 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Donchian channels (20-period)
    highest_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align to daily timeframe
    highest_20_aligned = align_htf_to_ltf(prices, df_1w, highest_20)
    lowest_20_aligned = align_htf_to_ltf(prices, df_1w, lowest_20)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume confirmation - 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20_aligned[i]) or np.isnan(lowest_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        highest_20_val = highest_20_aligned[i]
        lowest_20_val = lowest_20_aligned[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_avg = vol_avg_20[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > 1.5 * vol_avg if vol_avg > 0 else False
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price closes below weekly Donchian lower band OR trend turns bearish
            if (price < lowest_20_val) or (price < ema_34_val):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price closes above weekly Donchian upper band OR trend turns bullish
            if (price > highest_20_val) or (price > ema_34_val):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above weekly Donchian upper band + volume confirmation + bullish trend
            if (price > highest_20_val) and vol_confirm and (price > ema_34_val):
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below weekly Donchian lower band + volume confirmation + bearish trend
            elif (price < lowest_20_val) and vol_confirm and (price < ema_34_val):
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_TrendFilter"
timeframe = "1d"
leverage = 1.0