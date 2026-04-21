#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1w volume confirmation and 1d EMA trend filter.
# Captures trend continuation at key weekly high/low breakouts with volume surge.
# Works in both bull and bear markets by following institutional breakout zones.
# Target: 15-25 trades/year by requiring confluence of Donchian breakout, volume surge, and EMA trend alignment.
# Entry: Long when price breaks above 12h Donchian high(20) with volume spike and price > 1d EMA50; Short when breaks below Donchian low(20) with volume spike and price < 1d EMA50.
# Exit: Opposite Donchian touch or volume drops below average.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for volume and daily data for EMA
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 20-period EMA on daily timeframe
    close_d = df_1d['close'].values
    ema50_d = pd.Series(close_d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation using weekly volume
    vol_w = df_1w['volume'].values
    vol_ma_10_1w = pd.Series(vol_w).rolling(window=10, min_periods=10).mean().values
    
    # Align weekly and daily data to 12h (wait for weekly/daily close)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_d)
    vol_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10_1w)
    
    # Calculate 12h Donchian channels (20-period)
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
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_10_1w_aligned[i]) or 
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
        vol_current = align_htf_to_ltf(prices, df_1w, vol_w)[i]  # weekly volume aligned to 12h
        
        # Trend filter: price relative to daily EMA50
        above_ema = price_close > ema50_1d_aligned[i]
        below_ema = price_close < ema50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 10-week average
        volume_confirm = vol_current > 1.5 * vol_ma_10_1w_aligned[i]
        
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
                elif vol_current < vol_ma_10_1w_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price reaches Donchian high (opposite side) or volume drops below average
                if price_close > donchian_high[i]:
                    exit_signal = True
                elif vol_current < vol_ma_10_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian_Breakout_1wVolume_1dEMA50"
timeframe = "12h"
leverage = 1.0