#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ATR stoploss.
# Long when price breaks above 20-period high + volume > 1.5x 20-period avg.
# Short when price breaks below 20-period low + volume > 1.5x 20-period avg.
# Uses daily ATR for volatility-based stoploss. Designed for 12h timeframe to capture
# multi-day breakouts while avoiding false signals in choppy markets.
# Target: 20-40 trades/year per symbol (~80-160 total over 4 years).
name = "12h_Donchian20_Volume_ATRStop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ATR(14) on daily
    tr = np.zeros_like(high_1d)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                   abs(high_1d[i] - close_1d[i-1]), 
                   abs(low_1d[i] - close_1d[i-1]))
    
    atr_14 = np.zeros_like(tr)
    if len(tr) >= 15:
        atr_14[14] = np.mean(tr[1:15])
        for i in range(15, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align 1d ATR to 12h
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Donchian channels (20-period) on 12h
    donchian_high = np.full_like(high, np.nan)
    donchian_low = np.full_like(low, np.nan)
    
    for i in range(20, len(high)):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.full_like(volume, np.nan)
    vol_series = pd.Series(volume)
    vol_ma_20[20:] = vol_series.rolling(window=20, min_periods=20).mean().values[20:]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 20)  # Ensure Donchian and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        atr = atr_14_aligned[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above upper Donchian + volume confirmation
            if price > upper and volume_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Enter short if price breaks below lower Donchian + volume confirmation
            elif price < lower and volume_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: exit on Donchian low break or ATR-based stoploss
            if price < lower or price < entry_price - 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short position: exit on Donchian high break or ATR-based stoploss
            if price > upper or price > entry_price + 2.0 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals