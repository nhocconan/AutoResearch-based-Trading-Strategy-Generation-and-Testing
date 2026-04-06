#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Breakout with Daily EMA Trend Filter and Volume Confirmation
# Uses daily EMA200 to filter trend direction, 4-hour Donchian channel breakouts for entries,
# and volume confirmation (current volume > 1.8x 20-period average) to ensure quality.
# Works in both bull and bear markets: long only in uptrend (price > daily EMA200),
# short only in downtrend (price < daily EMA200). ATR-based stop loss limits drawdown.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_1d_ema200_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = np.full(len(close_1d), np.nan)
    
    # Calculate EMA200 on daily closes
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2/201) + (ema200_1d[i-1] * (1 - 2/201))
    
    # Align daily EMA200 to 4h timeframe (shifted by 1 daily bar)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 4-hour Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema200_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.8
        
        # Check exits and stoploss
        if position == 1:  # long position
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < donchian_low[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > donchian_high[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            if volume_filter:
                # Long: price > daily EMA200 (uptrend) and breakout above Donchian high
                if (close[i] > ema200_aligned[i] and 
                    close[i] > donchian_high[i] and 
                    close[i-1] <= donchian_high[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price < daily EMA200 (downtrend) and breakdown below Donchian low
                elif (close[i] < ema200_aligned[i] and 
                      close[i] < donchian_low[i] and 
                      close[i-1] >= donchian_low[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals