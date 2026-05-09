#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-week Donchian(10) breakout with 1-day EMA50 trend filter and volume confirmation.
# Enters long when price breaks above weekly Donchian high with daily uptrend and volume spike,
# short when price breaks below weekly Donchian low with daily downtrend and volume spike.
# Exits on trend reversal or price crossing opposite weekly level.
# Uses weekly timeframe for structure (less noisy) and daily for trend to avoid look-ahead.
# Designed to work in both bull and bear markets by aligning with daily trend.
# Target: 20-40 trades/year to minimize fee drag.

name = "4h_WeeklyDonchian10_1dEMA50_Volume"
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
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Donchian channels (10-period) on weekly high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian high: rolling max of high over 10 weeks
    donchian_high = pd.Series(high_1w).rolling(window=10, min_periods=10).max().values
    # Donchian low: rolling min of low over 10 weeks
    donchian_low = pd.Series(low_1w).rolling(window=10, min_periods=10).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on daily close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for EMA50 (1d) and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema50_1d_val = ema50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Close breaks above weekly Donchian high + daily uptrend + volume spike
            if close[i] > donchian_high_val and close[i] > ema50_1d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Close breaks below weekly Donchian low + daily downtrend + volume spike
            elif close[i] < donchian_low_val and close[i] < ema50_1d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below weekly Donchian low or daily trend turns down
            if close[i] < donchian_low_val or close[i] < ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above weekly Donchian high or daily trend turns up
            if close[i] > donchian_high_val or close[i] > ema50_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals