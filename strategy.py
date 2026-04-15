#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high + price > 12h EMA50 + volume > 1.5x 20-period avg volume
# Short when price breaks below 20-period Donchian low + price < 12h EMA50 + volume > 1.5x 20-period avg volume
# Uses 6h price structure with 12h EMA for multi-timeframe trend alignment
# Designed for low trade frequency (12-30/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and trend alignment via 12h EMA50

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # === 12h Indicators: EMA50 ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    for i in range(warmup, n):
        # Donchian(20) on 6h: highest high and lowest low of last 20 periods
        highest_20 = np.max(high[i-19:i+1]) if i >= 19 else np.nan
        lowest_20 = np.min(low[i-19:i+1]) if i >= 19 else np.nan
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        vol_confirm = volume[i] > (vol_sma_20 * 1.5) if not np.isnan(vol_sma_20) else False
        
        # Skip if any required data is NaN
        if (np.isnan(highest_20) or np.isnan(lowest_20) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma_20)):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. Price above 12h EMA50 (uptrend filter)
        # 3. Volume confirmation
        if (close[i] > highest_20) and (close[i] > ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. Price below 12h EMA50 (downtrend filter)
        # 3. Volume confirmation
        elif (close[i] < lowest_20) and (close[i] < ema_50_12h_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Donchian20_Volume_12hEMA50_Filter_v1"
timeframe = "6h"
leverage = 1.0