#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian Breakout with 4h Trend Filter and Volume Confirmation.
# Uses 1h Donchian channel (20-period) for entry timing, 4h EMA for trend direction,
# and volume spike (current volume > 1.5x 20-period average) for confirmation.
# Only trades in direction of 4h trend to avoid whipsaws. Designed for low trade frequency
# (target: 60-150 trades over 4 years) to minimize fee drag. Works in bull/bear via trend filter.

name = "1h_donchian_4h_trend_vol_v1"
timeframe = "1h"
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
    
    # 4h EMA for trend direction (calculated once, aligned to 1h)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1h Donchian channels (20-period high/low)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(donch_high[i]) or \
           np.isnan(donch_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend direction from 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian low or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] <= donch_low[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] >= donch_high[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume and trend confirmation
            if volume_filter:
                # Long: Donchian breakout + uptrend
                if (close[i] > donch_high[i] and close[i-1] <= donch_high[i] and uptrend):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: Donchian breakdown + downtrend
                elif (close[i] < donch_low[i] and close[i-1] >= donch_low[i] and downtrend):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals