#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 12h EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions. In strong trends (12h EMA slope),
# we take counter-trend entries at extremes. In weak trends, we avoid trading.
# Volume confirmation (>1.3x average) filters false signals.
# Designed to work in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
# Target: 50-150 total trades over 4 years (12-37/year).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    close_6h = df_6h['close'].values
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    volume_6h = df_6h['volume'].values
    
    # === 12h data (higher timeframe for trend filter) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # === Williams %R (14) on 6h ===
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    williams_r = williams_r.replace([np.inf, -np.inf], np.nan).fillna(0).values
    williams_r_aligned = align_htf_to_ltf(prices, df_6h, williams_r)
    
    # === 12h EMA(34) for trend filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # === 12h EMA(34) slope for trend strength ===
    ema_slope = np.diff(ema_34_12h_aligned, prepend=ema_34_12h_aligned[0])
    # Normalize slope by price to get dimensionless strength
    ema_slope_norm = ema_slope / (close_12h + 1e-10)  # avoid division by zero
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_slope_norm)
    
    # === 6h volume ratio for confirmation ===
    vol_ma_10_6h = pd.Series(volume_6h).rolling(window=10, min_periods=10).mean().values
    vol_ratio_6h = volume_6h / vol_ma_10_6h
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 50
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ratio_6h[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        wr = williams_r_aligned[i]
        ema_trend = ema_34_12h_aligned[i]
        ema_slope_val = ema_slope_aligned[i]
        vol_ratio = vol_ratio_6h[i]
        
        # === STOPLOSS LOGIC ===
        if position == 1:  # Long position
            # Stop loss: price closes below EMA34
            if price < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Stop loss: price closes above EMA34
            if price > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when Williams %R reaches overbought or trend weakens
            if wr > -20 or ema_slope_val < 0:  # Overbought or negative slope
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when Williams %R reaches oversold or trend weakens
            if wr < -80 or ema_slope_val > 0:  # Oversold or positive slope
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Strong uptrend: positive slope, buy oversold pullbacks
            if ema_slope_val > 0.0001 and wr < -80 and vol_ratio > 1.3:
                signals[i] = 0.25
                position = 1
                entry_price = price
                continue
            # Strong downtrend: negative slope, sell overbought rallies
            elif ema_slope_val < -0.0001 and wr > -20 and vol_ratio > 1.3:
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

name = "6h_WilliamsR_12hEMA34_Slope_Volume_v1"
timeframe = "6h"
leverage = 1.0