#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w volume confirmation and trend filter.
# Captures breakout of weekly structure with volume surge, avoiding false breakouts.
# Works in both bull and bear markets by requiring volume confirmation and trend alignment.
# Target: 10-20 trades/year by requiring confluence of Donchian breakout, volume surge (2x weekly average), and price above/below 1w EMA50.
# Entry: Long when price breaks above 1d Donchian high(20) with volume > 2x weekly average and price > 1w EMA50; Short when breaks below Donchian low(20) with volume > 2x weekly average and price < 1w EMA50.
# Exit: Opposite Donchian touch or volume drops below weekly average.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for volume and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly timeframe for trend filter
    close_w = df_1w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 10-period average volume on weekly timeframe for volume confirmation
    vol_w = df_1w['volume'].values
    vol_avg_10_w = pd.Series(vol_w).rolling(window=10, min_periods=10).mean().values
    
    # Align weekly data to 1d (wait for weekly close)
    ema50_w_aligned = align_htf_to_ltf(prices, df_1w, ema50_w)
    vol_avg_10_w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_10_w)
    
    # Calculate 1d Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_w_aligned[i]) or np.isnan(vol_avg_10_w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        vol_current = prices['volume'].iloc[i]
        
        # Trend filter: price relative to weekly EMA50
        above_ema = price_close > ema50_w_aligned[i]
        below_ema = price_close < ema50_w_aligned[i]
        
        # Volume confirmation: current volume > 2x 10-week average
        volume_confirm = vol_current > 2.0 * vol_avg_10_w_aligned[i]
        
        if position == 0:
            # Enter long when price breaks above Donchian high with volume surge and above weekly EMA
            if (price_close > donchian_high[i] and volume_confirm and above_ema):
                signals[i] = 0.25
                position = 1
            # Enter short when price breaks below Donchian low with volume surge and below weekly EMA
            elif (price_close < donchian_low[i] and volume_confirm and below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches Donchian low (opposite side) or volume drops below weekly average
                if price_close < donchian_low[i]:
                    exit_signal = True
                elif vol_current < vol_avg_10_w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price reaches Donchian high (opposite side) or volume drops below weekly average
                if price_close > donchian_high[i]:
                    exit_signal = True
                elif vol_current < vol_avg_10_w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian_Breakout_1wVolume_EMA50"
timeframe = "1d"
leverage = 1.0