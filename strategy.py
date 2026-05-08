#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1h RSI pullback for entry timing.
# Uses 4h Donchian channels (20-period) for trend direction, 1h RSI (14) for pullback entries in trend direction,
# and volume filter (1h volume > 1.5x 20-period EMA) to avoid false breakouts.
# Long when price breaks above 4h Donchian upper band, RSI < 40 (pullback), and volume confirmation.
# Short when price breaks below 4h Donchian lower band, RSI > 60 (pullback), and volume confirmation.
# Exit when price crosses 4h Donchian middle line or RSI reaches opposite extreme.
# Designed for 15-35 trades/year to avoid fee drag. Works in both trending and ranging markets via trend filter.

name = "1h_4hDonchian_RSI_Pullback"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    upper_channel = np.full_like(close_4h, np.nan)
    lower_channel = np.full_like(close_4h, np.nan)
    middle_channel = np.full_like(close_4h, np.nan)
    
    for i in range(20, len(close_4h)):
        upper_channel[i] = np.max(high_4h[i-19:i+1])
        lower_channel[i] = np.min(low_4h[i-19:i+1])
        middle_channel[i] = (upper_channel[i] + lower_channel[i]) / 2
    
    # Align 4h Donchian channels to 1h timeframe
    upper_chan_aligned = align_htf_to_ltf(prices, df_4h, upper_channel)
    lower_chan_aligned = align_htf_to_ltf(prices, df_4h, lower_channel)
    middle_chan_aligned = align_htf_to_ltf(prices, df_4h, middle_channel)
    
    # Calculate 1h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    # Wilder's smoothing
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan  # Not enough data
    
    # Volume confirmation: 1h volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for Donchian and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_chan_aligned[i]) or 
            np.isnan(lower_chan_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4h upper channel, RSI < 40 (pullback), volume confirmation
            if (close[i] > upper_chan_aligned[i] and 
                rsi[i] < 40 and 
                vol_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h lower channel, RSI > 60 (pullback), volume confirmation
            elif (close[i] < lower_chan_aligned[i] and 
                  rsi[i] > 60 and 
                  vol_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h middle channel or RSI > 60
            if close[i] < middle_chan_aligned[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h middle channel or RSI < 40
            if close[i] > middle_chan_aligned[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals