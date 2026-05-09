#!/usr/bin/env python3
name = "6H_RSI_Momentum_With_Volume_Confirmation"
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
    
    # Get weekly data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI14 for trend filter
    if len(close_1w) >= 14:
        delta = np.diff(close_1w)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_1w)
        avg_loss = np.zeros_like(close_1w)
        
        # First average
        avg_gain[13] = np.mean(gain[:13])
        avg_loss[13] = np.mean(loss[:13])
        
        # Wilder smoothing
        for i in range(14, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi_1w = 100 - (100 / (1 + rs))
        rsi_1w[:13] = np.nan
    else:
        rsi_1w = np.full_like(close_1w, np.nan)
    
    # Align weekly RSI to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Get daily data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volume EMA20
    if len(volume_1d) >= 20:
        vol_ema20_1d = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    else:
        vol_ema20_1d = np.full_like(volume_1d, np.nan)
    
    # Align daily volume EMA20 to 6h timeframe
    vol_ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for all indicators
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(rsi_1w_aligned[i]) or np.isnan(vol_ema20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market conditions
        # Weekly RSI > 50 = bullish bias, < 50 = bearish bias
        bullish_bias = rsi_1w_aligned[i] > 50
        bearish_bias = rsi_1w_aligned[i] < 50
        # Volume surge: current volume > 1.5x daily volume EMA20
        volume_surge = volume[i] > vol_ema20_1d_aligned[i] * 1.5
        
        if position == 0:
            # Enter long: Bullish bias + price above weekly RSI 50 level + volume surge
            if bullish_bias and close[i] > close_1w[-1] if len(close_1w) > 0 else False and volume_surge:
                signals[i] = 0.25
                position = 1
            # Enter short: Bearish bias + price below weekly RSI 50 level + volume surge
            elif bearish_bias and close[i] < close_1w[-1] if len(close_1w) > 0 else False and volume_surge:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bearish bias OR volume drops below average
            if bearish_bias or volume[i] <= vol_ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bullish bias OR volume drops below average
            if bullish_bias or volume[i] <= vol_ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals