#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA50 trend filter
# Long when price breaks above 20-period Donchian high + volume > 1.8x 20-period volume SMA + price > 12h EMA50
# Short when price breaks below 20-period Donchian low + volume > 1.8x 20-period volume SMA + price < 12h EMA50
# Uses Donchian channels for price structure and 12h EMA for trend alignment on 4h chart
# Designed for low trade frequency (20-50/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

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
        # Donchian(20) channels
        donchian_high = np.max(high[i-19:i+1]) if i >= 19 else np.nan
        donchian_low = np.min(low[i-19:i+1]) if i >= 19 else np.nan
        
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_sma_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.nan
        vol_confirm = volume[i] > (vol_sma_20 * 1.8) if not np.isnan(vol_sma_20) else False
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high) or np.isnan(donchian_low) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma_20)):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. Volume confirmation
        # 3. Price above 12h EMA50 (uptrend filter)
        if (close[i] > donchian_high) and vol_confirm and (close[i] > ema_50_12h_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. Volume confirmation
        # 3. Price below 12h EMA50 (downtrend filter)
        elif (close[i] < donchian_low) and vol_confirm and (close[i] < ema_50_12h_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_12hEMA50_Filter_v1"
timeframe = "4h"
leverage = 1.0