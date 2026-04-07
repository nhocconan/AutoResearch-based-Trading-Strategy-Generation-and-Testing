#!/usr/bin/env python3
"""
4h_donchian_20_12h_trend_volume_v3
Hypothesis: On 4-hour timeframe, buy breakouts above 20-period Donchian channel when 12h trend is up (close > EMA50) and volume confirms; sell breakdowns below 20-period Donchian channel when 12h trend is down (close < EMA50) and volume confirms. Uses volatility filter (ATR) to avoid choppy markets. Designed for 20-50 trades/year to minimize fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_12h_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # Calculate ATR(14) on 4h for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_period = 14
    atr = np.full_like(tr, np.nan)
    for i in range(len(tr)):
        if np.isnan(tr[i]):
            if i == 0:
                atr[i] = np.nan
            else:
                atr[i] = atr[i-1]
        else:
            if i == 0 or np.isnan(atr[i-1]):
                atr[i] = tr[i]
            else:
                atr[i] = atr[i-1] + (1/atr_period) * (tr[i] - atr[i-1])
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # Calculate average volume (20-period)
    def rolling_mean(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i < window - 1:
                res[i] = np.nan
            else:
                res[i] = np.mean(arr[i-window+1:i+1])
        return res
    
    avg_volume = rolling_mean(volume, 20)
    
    # Align 12h indicators to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    atr_aligned = align_htf_to_ltf(prices, df_12h, atr)  # Use 12h ATR for volatility filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any data not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(atr_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility (choppy) markets
        if atr_aligned[i] < 0.5 * np.nanmean(atr_aligned[max(0, i-50):i+1]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price re-enters Donchian channel or trend changes
            if close[i] <= donch_high[i] or close[i] < ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price re-enters Donchian channel or trend changes
            if close[i] >= donch_low[i] or close[i] > ema50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: breakout above Donchian high with up trend and volume
            if (close[i] > donch_high[i] and 
                close[i] > ema50_12h_aligned[i] and 
                volume_confirmed):
                position = 1
                signals[i] = 0.25
            # Short: breakdown below Donchian low with down trend and volume
            elif (close[i] < donch_low[i] and 
                  close[i] < ema50_12h_aligned[i] and 
                  volume_confirmed):
                position = -1
                signals[i] = -0.25
    
    return signals