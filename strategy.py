#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R combined with 1d EMA filter and volume confirmation.
# Uses Williams %R(14) on 12h for overbought/oversold signals.
# In oversold (%R < -80): look for long entries when price > 1d EMA(34) and volume > 1.5x average.
# In overbought (%R > -20): look for short entries when price < 1d EMA(34) and volume > 1.5x average.
# EMA filter ensures trading with the higher timeframe trend.
# Volume confirmation reduces false signals.
# Designed to work in both bull (buy oversold in uptrend) and bear (sell overbought in downtrend).

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
    
    # === 1d data (higher timeframe for EMA filter) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 12h Williams %R (14) ===
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_12h) / (highest_high - lowest_low) * -100
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(-50).values
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # === 1d EMA(34) for trend filter ===
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 12h volume ratio for confirmation ===
    vol_ma_10_12h = pd.Series(volume_12h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_12h = volume_12h / vol_ma_10_12h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ratio_12h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        ema_34 = ema_34_1d_aligned[i]
        vol_ratio = vol_ratio_12h[i]
        
        # === STOPLOSS LOGIC ===
        # Calculate ATR-based stop using 12h data
        if i >= 14:  # Need enough data for ATR
            tr1 = high_12h[i] - low_12h[i]
            tr2 = abs(high_12h[i] - close_12h[i-1]) if i-1 >= 0 else 0
            tr3 = abs(low_12h[i] - close_12h[i-1]) if i-1 >= 0 else 0
            tr = max(tr1, tr2, tr3)
            # Simplified ATR calculation for stop (using current bar's TR)
            atr_est = tr  # Using current bar's true range as proxy
            
            if position == 1:  # Long position
                if price < entry_price - 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
            
            elif position == -1:  # Short position
                if price > entry_price + 2.0 * atr_est:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                    continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Williams %R returns to overbought or EMA breaks down
            if wr > -20 or price < ema_34:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns to oversold or EMA breaks up
            if wr < -80 or price > ema_34:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            if wr < -80 and price > ema_34 and vol_ratio > 1.5:  # Oversold + above EMA + volume
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            elif wr > -20 and price < ema_34 and vol_ratio > 1.5:  # Overbought + below EMA + volume
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

name = "12h_WilliamsR_1dEMA34_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0