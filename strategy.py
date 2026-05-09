#!/usr/bin/env python3
name = "6H_Anchored_VWAP_Slope_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for weekly trend filter and VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily VWAP from the start of the week (Monday)
    # We'll anchor VWAP to weekly start by using cumulative from Monday
    # For simplicity, we use daily VWAP as a proxy and smooth it
    # In practice, weekly anchored VWAP would use data from Monday to current day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_num = (typical_price * df_1d['volume']).cumsum()
    vwap_den = df_1d['volume'].cumsum()
    vwap = vwap_num / vwap_den
    vwap = vwap.values
    
    # Calculate VWAP slope (5-period change)
    if len(vwap) >= 6:
        vwap_slope = (vwap[5:] - vwap[:-5]) / 5  # 5-period slope
        # Pad beginning with NaN
        vwap_slope = np.concatenate([np.full(5, np.nan), vwap_slope])
    else:
        vwap_slope = np.full_like(vwap, np.nan)
    
    # Align daily VWAP slope to 6h timeframe
    vwap_slope_aligned = align_htf_to_ltf(prices, df_1d, vwap_slope)
    
    # Get 6h data for price position and volume filter
    # Calculate 6-period RSI for momentum filter
    if len(close) >= 14:
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = np.full_like(close, 50.0)
    
    # Volume filter: current volume > 1.3x 20-period average
    if len(volume) >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma = np.full_like(volume, np.nan)
    volume_surge = volume > vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = max(14, 20)  # RSI and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_slope_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine conditions
        # VWAP slope positive = bullish bias, negative = bearish bias
        vwap_bullish = vwap_slope_aligned[i] > 0
        vwap_bearish = vwap_slope_aligned[i] < 0
        # RSI not extreme (avoid chasing)
        rsi_not_overbought = rsi[i] < 70
        rsi_not_oversold = rsi[i] > 30
        # Volume confirmation
        vol_confirm = volume_surge[i]
        
        if position == 0:
            # Enter long: VWAP slope bullish + RSI not overbought + volume surge
            if vwap_bullish and rsi_not_overbought and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: VWAP slope bearish + RSI not oversold + volume surge
            elif vwap_bearish and rsi_not_oversold and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: VWAP slope turns bearish OR RSI overbought
            if not vwap_bullish or rsi[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: VWAP slope turns bullish OR RSI oversold
            if not vwap_bearish or rsi[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals