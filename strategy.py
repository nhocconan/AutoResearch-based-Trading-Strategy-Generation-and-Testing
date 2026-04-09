#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly trend filter and volume confirmation
# Long when price breaks above 20-day high with weekly EMA21 uptrend and volume > 2x average
# Short when price breaks below 20-day low with weekly EMA21 downtrend and volume > 2x average
# Weekly EMA21 determines trend direction to avoid counter-trend trades
# Position size 0.25 to limit drawdown, targeting ~15-25 trades/year for low fee drag
# Designed to work in both bull (trend following) and bear (counter-trend reversals at extremes)

name = "1d_1w_donchian_trend_v2"
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
    
    # Calculate weekly EMA21 for trend filter
    close_1w = df_1w['close'].values
    ema_21 = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 21:
        ema_21[20] = np.mean(close_1w[:21])
        for i in range(21, len(close_1w)):
            ema_21[i] = (close_1w[i] * 2/22) + (ema_21[i-1] * 20/22)
    
    # Align weekly EMA21 to daily timeframe
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Daily Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 20-period average
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
            np.isnan(ema_21_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA21 OR donch_low breaks (trend change)
            if close[i] < ema_21_aligned[i] or close[i] < donch_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA21 OR donch_high breaks
            if close[i] > ema_21_aligned[i] or close[i] > donch_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above donch_high with weekly uptrend and volume confirmation
            vol_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 0 else 0
            if (close[i] > donch_high[i] and 
                ema_21_aligned[i] > ema_21_aligned[i-1] and  # Weekly EMA rising
                vol_ratio > 2.0):
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below donch_low with weekly downtrend and volume confirmation
            elif (close[i] < donch_low[i] and 
                  ema_21_aligned[i] < ema_21_aligned[i-1] and  # Weekly EMA falling
                  vol_ratio > 2.0):
                position = -1
                signals[i] = -0.25
    
    return signals