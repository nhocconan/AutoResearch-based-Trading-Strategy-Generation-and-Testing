#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian breakout with daily RSI filter and volume confirmation.
# Uses weekly price structure for trend context and daily momentum for entry timing.
# Long when price breaks above weekly Donchian high (20) and daily RSI > 55 with volume confirmation.
# Short when price breaks below weekly Donchian low (20) and daily RSI < 45 with volume confirmation.
# Exit when price returns to weekly Donchian midpoint or RSI reverses.
# Designed for low trade frequency (10-20/year) to avoid fee decay. Works in trending markets via trend filter.

name = "1d_20wDonchian_RSI_Volume"
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
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 20-week Donchian channels
    highest_high = np.full_like(high_1w, np.nan)
    lowest_low = np.full_like(low_1w, np.nan)
    
    for i in range(19, len(high_1w)):
        highest_high[i] = np.max(high_1w[i-19:i+1])
        lowest_low[i] = np.min(low_1w[i-19:i+1])
    
    # Calculate midpoint
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 14-day RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = np.mean(gain[:1]) if len(gain[:1]) > 0 else 0
                avg_loss[i] = np.mean(loss[:1]) if len(loss[:1]) > 0 else 0
            else:
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan  # Not enough data for first 14 periods
    
    # Align indicators to daily timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: daily volume > 1.5x 20-day EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high_aligned[i]) or 
            np.isnan(lowest_low_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly Donchian high and RSI > 55 with volume confirmation
            if (close[i] > highest_high_aligned[i] and 
                rsi_aligned[i] > 55 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly Donchian low and RSI < 45 with volume confirmation
            elif (close[i] < lowest_low_aligned[i] and 
                  rsi_aligned[i] < 45 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to weekly Donchian midpoint or RSI < 45
            if close[i] < donchian_mid_aligned[i] or rsi_aligned[i] < 45:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to weekly Donchian midpoint or RSI > 55
            if close[i] > donchian_mid_aligned[i] or rsi_aligned[i] > 55:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals