#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and 1d RSI filter.
# Long when price breaks above upper Donchian channel + volume spike + 1d RSI < 70 (avoid overbought)
# Short when price breaks below lower Donchian channel + volume spike + 1d RSI > 30 (avoid oversold)
# Exit when price returns to middle of Donchian channel.
# Works in trending markets (breakouts) and avoids counter-trend entries.
# Target: 20-40 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily close
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    
    # Calculate Donchian channels (20-period high/low) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle = (upper + lower) / 2
    
    # Align 1d RSI to 4h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or 
            np.isnan(lower[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi = rsi_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian + volume spike + RSI not overbought
            if price > upper[i] and vol_spike and rsi < 70:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian + volume spike + RSI not oversold
            elif price < lower[i] and vol_spike and rsi > 30:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to middle of Donchian channel
            if abs(price - middle[i]) < (upper[i] - lower[i]) * 0.1:  # within 10% of middle
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Volume_RSI"
timeframe = "4h"
leverage = 1.0