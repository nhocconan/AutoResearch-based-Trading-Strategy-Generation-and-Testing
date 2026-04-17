#!/usr/bin/env python3
"""
Hypothesis: In BTC/ETH markets, price tends to respect the Volume Weighted Average Price (VWAP) 
as a dynamic equilibrium. Combining 1-day VWAP with 4-hour Bollinger Band breakouts and volume 
confirmation creates high-probability mean-reversion trades. The strategy targets 20-30 trades 
per year by requiring price to deviate significantly from VWAP (>2 standard deviations) with 
volume expansion, then revert back to VWAP. Works in both bull (buy dips to VWAP) and bear 
(sell rallies to VWAP) regimes by fading extremes.
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
    
    # Get daily data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily VWAP and standard deviation
    # VWAP = sum(price * volume) / sum(volume) over the day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    
    # Calculate daily standard deviation of price from VWAP
    price_dev = typical_price - vwap
    variance = (price_dev ** 2 * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    std_dev = np.sqrt(variance)
    
    # Bollinger Bands around VWAP: VWAP ± 2 * std_dev
    upper_band = vwap + 2.0 * std_dev
    lower_band = vwap - 2.0 * std_dev
    
    # Align all daily levels to 4h timeframe (waits for daily bar to close)
    vwap_4h = align_htf_to_ltf(prices, df_1d, vwap.values)
    upper_4h = align_htf_to_ltf(prices, df_1d, upper_band.values)
    lower_4h = align_htf_to_ltf(prices, df_1d, lower_band.values)
    
    # Volume confirmation: 20-period volume MA on 4h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 40  # warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_4h[i]) or np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or
            np.isnan(volume_ma_20.iloc[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = volume_ma_20.iloc[i]
        
        if position == 0:
            # Long: price below lower Bollinger Band with volume expansion
            if price < lower_4h[i] and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price above upper Bollinger Band with volume expansion
            elif price > upper_4h[i] and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to VWAP or volume dries up
            if price > vwap_4h[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or volume dries up
            if price < vwap_4h[i] or vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_BBands_Volume_MeanReversion"
timeframe = "4h"
leverage = 1.0