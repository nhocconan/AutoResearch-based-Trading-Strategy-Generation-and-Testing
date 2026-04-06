#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian Breakout with 12h Trend Filter and Volume Confirmation.
# Uses Donchian channel (20) breakout on 4h timeframe, filtered by 12h EMA trend direction.
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# ATR-based stoploss (2x ATR) manages risk. Works in bull/bear markets via trend filter.
# Target: 75-200 trades over 4 years (19-50/year).

name = "4h_donchian20_12h_ema_vol_v1"
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
    
    # Calculate ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.full(n, np.nan)
    for i in range(13, n):
        atr[i] = np.nanmean(tr[i-13:i+1])  # ATR(14)
    
    # Donchian channel (20-period)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(19, n):
        upper[i] = np.max(high[i-19:i+1])
        lower[i] = np.min(low[i-19:i+1])
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    for i in range(1, len(close_12h)):
        ema_12h[i] = 0.2 * close_12h[i] + 0.8 * ema_12h[i-1] if not np.isnan(ema_12h[i-1]) else close_12h[i]
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches lower Donchian or stoploss
            stop_loss_level = entry_price - 2.0 * atr[i]
            
            if (close[i] <= lower[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches upper Donchian or stoploss
            stop_loss_level = entry_price + 2.0 * atr[i]
            
            if (close[i] >= upper[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend filter
            if volume_filter:
                # Long breakout above upper Donchian with 12h uptrend
                if (close[i] > upper[i] and close[i-1] <= upper[i] and 
                    close[i] > ema_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short breakdown below lower Donchian with 12h downtrend
                elif (close[i] < lower[i] and close[i-1] >= lower[i] and 
                      close[i] < ema_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals