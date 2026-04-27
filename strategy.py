#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter, volume confirmation, and ATR stop
# Donchian breakouts capture momentum; 1d trend filters direction; volume confirms strength.
# Works in bull/bear by only taking breakouts in direction of 1d trend.
# Target: 100-200 total trades over 4 years (~25-50/year) to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA trend filter (50-period)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # ATR for stop loss (14-period)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr[i] = np.mean(tr[i-13:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 1d EMA (50), volume MA (20), ATR (14)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        atr_now = atr[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Trend filter from 1d EMA
        bullish_trend = price > ema_50_aligned[i]
        bearish_trend = price < ema_50_aligned[i]
        
        if position == 0:
            # Donchian breakout calculation (20-period)
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                donchian_low = np.min(low[i-20:i])
                
                # Long: break above Donchian high with volume and bullish trend
                if price > donchian_high and vol_filter and bullish_trend:
                    signals[i] = size
                    position = 1
                # Short: break below Donchian low with volume and bearish trend
                elif price < donchian_low and vol_filter and bearish_trend:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below Donchian low or trend turns bearish
            if i >= 20:
                donchian_low = np.min(low[i-20:i])
                if price < donchian_low or not bullish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = size
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above Donchian high or trend turns bullish
            if i >= 20:
                donchian_high = np.max(high[i-20:i])
                if price > donchian_high or not bearish_trend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -size
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0