#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using Donchian breakout with 1d RSI filter and volume spike.
- Calculate Donchian channels (20-period high/low) on 4h data
- Enter long when price breaks above upper band with volume > 1.5x 20-period volume MA and 1d RSI > 50
- Enter short when price breaks below lower band with volume > 1.5x 20-period volume MA and 1d RSI < 50
- Exit when price crosses back to the opposite Donchian band
- Fixed position size 0.25 to manage drawdown
- Uses 1d RSI to avoid counter-trend trades in sideways markets
"""

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
    
    # Get 1-day data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-period RSI on 1d closes
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_14_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation: 20-period volume MA
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high.iloc[i]) or 
            np.isnan(donchian_low.iloc[i]) or 
            np.isnan(volume_ma_20.iloc[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        upper = donchian_high.iloc[i]
        lower = donchian_low.iloc[i]
        rsi_val = rsi_1d_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts with volume confirmation and RSI filter
            # Long: price breaks above upper band + volume spike + RSI > 50 (bullish)
            if price > upper and vol > 1.5 * vol_ma and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band + volume spike + RSI < 50 (bearish)
            elif price < lower and vol > 1.5 * vol_ma and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price crosses below lower band (opposite side)
            if price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price crosses above upper band (opposite side)
            if price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_Volume_1dRSI"
timeframe = "4h"
leverage = 1.0