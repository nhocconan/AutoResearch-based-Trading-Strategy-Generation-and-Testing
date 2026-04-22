#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + Donchian(20) breakout with volume confirmation.
# In trending regimes (CHOP < 38.2), trade breakouts in direction of trend.
# In ranging regimes (CHOP > 61.8), fade reversals at Donchian bands.
# Uses 12h trend filter (EMA50) to avoid counter-trend trades.
# Designed for low trade frequency (<30/year) to minimize fee drag in choppy markets like 2025.
# Works in bull (trend breakouts) and bear (mean reversion in range) markets.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Choppiness Index (14-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    # ATR(14)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(tr_sum / (atr * 14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((atr * 14) > 0, chop, 50.0)
    
    # Donchian channels (20-period)
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(chop[i]) or 
            np.isnan(dc_high[i]) or 
            np.isnan(dc_low[i]) or 
            np.isnan(ema50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        chop_val = chop[i]
        upper = dc_high[i]
        lower = dc_low[i]
        trend = ema50_12h_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            if chop_val < 38.2:  # Trending regime
                # Long: price breaks above Donchian high + volume spike + above 12h EMA50
                if price > upper and vol_spike and price > trend:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below Donchian low + volume spike + below 12h EMA50
                elif price < lower and vol_spike and price < trend:
                    signals[i] = -0.25
                    position = -1
            elif chop_val > 61.8:  # Ranging regime
                # Long: price reverses up from Donchian low + volume spike
                if price < lower * 1.005 and price > lower and vol_spike:
                    signals[i] = 0.25
                    position = 1
                # Short: price reverses down from Donchian high + volume spike
                elif price > upper * 0.995 and price < upper and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                if chop_val < 38.2:  # trending: exit on Donchian low break
                    if price < lower:
                        exit_signal = True
                else:  # ranging: exit on Donchian high touch or volume dry
                    if price > upper * 0.995 or vol < 0.8 * vol_ma:
                        exit_signal = True
            
            elif position == -1:  # short position
                if chop_val < 38.2:  # trending: exit on Donchian high break
                    if price > upper:
                        exit_signal = True
                else:  # ranging: exit on Donchian low touch or volume dry
                    if price < lower * 1.005 or vol < 0.8 * vol_ma:
                        exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Chop_Donchian_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0