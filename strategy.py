#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation (>1.5x average) + ATR(14) stoploss.
Long when price breaks above Donchian upper band with HMA up and volume spike.
Short when price breaks below Donchian lower band with HMA down and volume spike.
Exit via ATR-based trailing stop: signal=0 when price < highest high since entry - 2.5*ATR (long) or
price > lowest low since entry + 2.5*ATR (short). Uses 1d for volume average to reduce noise.
Target: 75-200 total trades over 4 years (19-50/year). Discrete sizing: 0.25.
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
    
    # Get 1d data for volume average (less noisy than 4h)
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(low, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    upper_4h, lower_4h = donchian_channels(high, low, 20)
    
    # Calculate 4h HMA(21) for trend filter
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).rolling(window=half, min_periods=half).mean().values
        wma1 = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).rolling(window=sqrt, min_periods=sqrt).mean().values
        return hma_vals
    
    hma_4h = hma(close, 21)
    
    # Calculate 14-period ATR for stoploss
    def atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr[0] = high[0] - low[0]
        atr_vals = np.zeros_like(close)
        atr_vals[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr_14 = atr(high, low, close, 14)
    
    # Calculate 1d volume average (20-period) for spike detection
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Align 4h indicators to 4h timeframe (self-aligning)
    upper_4h_aligned = align_htf_to_ltf(prices, prices, upper_4h)  # self-align
    lower_4h_aligned = align_htf_to_ltf(prices, prices, lower_4h)
    hma_4h_aligned = align_htf_to_ltf(prices, prices, hma_4h)
    atr_14_aligned = align_htf_to_ltf(prices, prices, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_4h_aligned[i]) or 
            np.isnan(lower_4h_aligned[i]) or 
            np.isnan(hma_4h_aligned[i]) or 
            np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume[i] > (vol_ma_1d_aligned[i] * 1.5)
        upper = upper_4h_aligned[i]
        lower = lower_4h_aligned[i]
        hma = hma_4h_aligned[i]
        atr_val = atr_14_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with HMA up and volume spike
            if price > upper and hma > hma_4h_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: price breaks below lower Donchian with HMA down and volume spike
            elif price < lower and hma < hma_4h_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops below highest - 2.5*ATR
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises above lowest + 2.5*ATR
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_HMA21_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0