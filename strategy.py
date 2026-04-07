#!/usr/bin/env python3
"""
6h_wick_reversal_volume_v1
Hypothesis: On 6h timeframe, enter long when price rejects lower wick (close near high) with volume spike during uptrend (price above 12h EMA50), enter short when price rejects upper wick (close near low) with volume spike during downtrend (price below 12h EMA50). Uses 12h EMA50 trend filter to avoid counter-trend trades. Target: 20-40 trades/year to minimize fee drag while capturing rejection patterns in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_wick_reversal_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Wick rejection signals
    body_size = np.abs(close - open_price)
    total_range = high - low
    lower_wick = np.where(close >= open_price, close - open_price, open_price - close)  # bullish body
    upper_wick = np.where(close <= open_price, open_price - close, close - open_price)  # bearish body
    
    # Long rejection: close near high (small upper wick)
    # Short rejection: close near low (small lower wick)
    upper_wick_ratio = np.where(total_range > 0, upper_wick / total_range, 0)
    lower_wick_ratio = np.where(total_range > 0, lower_wick / total_range, 0)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if data not available
        if (np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(open_price[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        # Wick rejection conditions
        long_rejection = (upper_wick_ratio[i] < 0.3) and (close[i] > open_price[i])  # close near high, bullish candle
        short_rejection = (lower_wick_ratio[i] < 0.3) and (close[i] < open_price[i])  # close near low, bearish candle
        
        if position == 1:  # Long position
            # Exit: rejection fails or trend changes
            if not long_rejection or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: rejection fails or trend changes
            if not short_rejection or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: bullish rejection in uptrend
                if long_rejection and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: bearish rejection in downtrend
                elif short_rejection and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals