#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Uses weekly EMA(21) as trend filter to ensure we trade in direction of higher timeframe trend
# Enters on daily Donchian breakout with volume > 1.5x 20-day average
# Exits when price closes opposite Donchian band
# Position size 0.25 to limit drawdown
# Target: 15-25 trades/year per symbol to minimize fee drag

name = "1d_1w_donchian_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = np.full(len(df_1w), np.nan)
    
    # Calculate EMA with proper seeding
    ema_21_1w[20] = np.mean(close_1w[0:21])  # Seed with simple average
    for i in range(21, len(df_1w)):
        ema_21_1w[i] = (close_1w[i] * 2 + ema_21_1w[i-1] * 19) / 21
    
    # Align weekly EMA to daily timeframe (only use completed weekly bars)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate daily Donchian channel (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: 20-day average volume
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0.0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below daily Donchian low
            if close[i] <= donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above daily Donchian high
            if close[i] >= donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price closes above daily Donchian high with volume confirmation and weekly uptrend
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donch_high[i] and 
                vol_ratio > 1.5 and
                close[i] > ema_21_1w_aligned[i]):  # Weekly uptrend filter
                position = 1
                signals[i] = 0.25
            # Enter short: price closes below daily Donchian low with volume confirmation and weekly downtrend
            elif (close[i] < donch_low[i] and 
                  vol_ratio > 1.5 and
                  close[i] < ema_21_1w_aligned[i]):  # Weekly downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals