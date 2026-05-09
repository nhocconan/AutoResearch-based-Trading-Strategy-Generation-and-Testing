#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and 1d RSI filter.
# Uses 1d average volume to filter breakouts (volume > 1.5x average) and 1d RSI to avoid
# extremes (40 < RSI < 60). This reduces false breakouts in low-volume or overextended
# conditions. Designed to work in both bull and bear markets by focusing on
# high-probability breakouts with volume confirmation. Target: 20-40 trades/year.
name = "4h_Donchian20_1dVolume_RSI_Filter"
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
    
    # Get 1d data for volume average and RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period average volume on 1d
    vol_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    # Calculate 14-period RSI on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    if len(gain) > 0:
        avg_gain[0] = gain[0]
        avg_loss[0] = loss[0]
        for i in range(1, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need 20 periods for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(avg_vol_20_aligned[i]) or np.isnan(rsi_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        price = close[i]
        vol_now = volume[i]
        avg_vol = avg_vol_20_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above 20-period high AND volume > 1.5x avg AND RSI in neutral range
            if price > highest_high and vol_now > 1.5 * avg_vol and 40 < rsi_val < 60:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 20-period low AND volume > 1.5x avg AND RSI in neutral range
            elif price < lowest_low and vol_now > 1.5 * avg_vol and 40 < rsi_val < 60:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 20-period low OR RSI > 60 (overbought)
            if price < lowest_low or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 20-period high OR RSI < 40 (oversold)
            if price > highest_high or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals