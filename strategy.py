#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high + volume > 1.5x 20-day volume average + price > 1w EMA50
# Short when price breaks below 20-day Donchian low + volume > 1.5x 20-day volume average + price < 1w EMA50
# Uses 1d price structure (Donchian channels) and 1w EMA for trend alignment
# Designed for very low trade frequency (7-25/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and higher-timeframe trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w Indicators: EMA50 ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    for i in range(warmup, n):
        # Donchian(20) channels on 1d
        donchian_high = np.max(high[i-19:i+1])  # 20-period high including current
        donchian_low = np.min(low[i-19:i+1])    # 20-period low including current
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = np.mean(volume[i-19:i+1]) if i >= 19 else np.nan
        vol_confirm = volume[i] > (vol_sma_20 * 1.5) if not np.isnan(vol_sma_20) else False
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high) or np.isnan(donchian_low) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20)):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-day Donchian high
        # 2. Volume confirmation
        # 3. Price above 1w EMA50 (uptrend filter)
        if (close[i] > donchian_high) and vol_confirm and (close[i] > ema_50_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-day Donchian low
        # 2. Volume confirmation
        # 3. Price below 1w EMA50 (downtrend filter)
        elif (close[i] < donchian_low) and vol_confirm and (close[i] < ema_50_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Donchian20_Volume_1wEMA50_Filter_v1"
timeframe = "1d"
leverage = 1.0