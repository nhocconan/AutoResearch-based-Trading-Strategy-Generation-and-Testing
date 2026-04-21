#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d volume confirmation and 1w EMA trend filter.
# Captures trend continuation at key weekly EMA-aligned breakouts with volume surge.
# Works in both bull and bear markets by following institutional breakout zones.
# Target: 20-30 trades/year by requiring confluence of Donchian breakout, volume surge, and EMA trend alignment.
# Entry: Long when price breaks above 4h Donchian high(20) with volume spike and price > 1w EMA50; Short when breaks below Donchian low(20) with volume spike and price < 1w EMA50.
# Exit: Opposite Donchian touch or volume drops below average.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for EMA and daily data for volume
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly timeframe
    close_w = df_1w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation using daily volume
    vol_d = df_1d['volume'].values
    vol_ma_10_1d = pd.Series(vol_d).rolling(window=10, min_periods=10).mean().values
    
    # Align weekly and daily data to 4h (wait for weekly/daily close)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_w)
    vol_ma_10_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_10_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_10_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current values
        price_close = prices['close'].iloc[i]
        vol_current = prices['volume'].iloc[i]
        
        # Trend filter: price relative to weekly EMA50
        above_ema = price_close > ema50_1w_aligned[i]
        below_ema = price_close < ema50_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 10-day average
        volume_confirm = vol_current > 1.5 * vol_ma_10_1d_aligned[i]
        
        if position == 0:
            # Enter long when price breaks above Donchian high with volume spike and above EMA
            if (price_close > donchian_high[i] and volume_confirm and above_ema):
                signals[i] = 0.25
                position = 1
            # Enter short when price breaks below Donchian low with volume spike and below EMA
            elif (price_close < donchian_low[i] and volume_confirm and below_ema):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches Donchian low (opposite side) or volume drops below average
                if price_close < donchian_low[i]:
                    exit_signal = True
                elif vol_current < vol_ma_10_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price reaches Donchian high (opposite side) or volume drops below average
                if price_close > donchian_high[i]:
                    exit_signal = True
                elif vol_current < vol_ma_10_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian_Breakout_1dVolume_1wEMA50"
timeframe = "4h"
leverage = 1.0