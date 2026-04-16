#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with 1d Trend Filter and Volume Spike
# Williams %R identifies overbought/oversold conditions (-20 to -80 range).
# Long when %R < -80 (oversold) and price above 1d EMA50 with volume spike.
# Short when %R > -20 (overbought) and price below 1d EMA50 with volume spike.
# Uses 1d EMA50 for trend filter to avoid counter-trend trades.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (higher timeframe for trend filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # === Williams %R on 12h ===
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Typically uses 14 period
    lookback = 14
    highest_high = pd.Series(high_12h).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_12h).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close_12h) / (highest_high - lowest_low)) * -100, 
                          -50)  # neutral when range is zero
    
    # === 1d EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h volume spike detection ===
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_12h > (2.0 * vol_ma_20_12h)  # stricter volume requirement
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        ema50 = ema50_1d_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price < entry_price - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            atr_12h = np.abs(high_12h - low_12h)
            atr_ma = pd.Series(atr_12h).rolling(window=14, min_periods=14).mean().values
            atr_aligned = align_htf_to_ltf(prices, df_12h, atr_ma)
            atr_val = atr_aligned[i]
            if price > entry_price + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Williams %R rises above -50 (momentum fading)
            if wr > -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum fading)
            if wr < -50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume spike
            if vol_spike_val:
                # Go long when oversold (%R < -80) and price above 1d EMA50
                if wr < -80 and price > ema50:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # Go short when overbought (%R > -20) and price below 1d EMA50
                elif wr > -20 and price < ema50:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_WilliamsR_1dEMA50_Volume_Spike"
timeframe = "12h"
leverage = 1.0