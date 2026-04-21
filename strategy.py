#!/usr/bin/env python3
"""
1d_Adaptive_Volume_Trend_Filter_v1
Hypothesis: On daily timeframe, use 20-period EMA for trend direction and volume spike (2.0x 20-period MA) for entry confirmation. Only take long positions when price > EMA20 and volume spike, short when price < EMA20 and volume spike. Use 1-week HTF trend filter (price > weekly EMA20 for longs, < weekly EMA20 for shorts) to avoid counter-trend trades. ATR-based stop loss (2.5x) and minimum holding period of 3 days to reduce churn. Designed to work in both bull and bear markets by aligning with higher timeframe trend and requiring strong volume confirmation. Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === 1d EMA20 for trend direction ===
    close = prices['close'].values
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 1d ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (2.0x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1w EMA20 for HTF trend filter ===
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_20_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_20_val = ema_20[i]
        vol_avg = vol_ma[i]
        ema_20_1w_val = ema_20_1w_aligned[i]
        
        # Volume confirmation: current volume > 2.0x average
        volume_confirm = volume_now > 2.0 * vol_avg
        
        if position == 0:
            # Long: price above 1d EMA20, above 1w EMA20, volume confirm
            long_condition = (price > ema_20_val) and (price > ema_20_1w_val) and volume_confirm
            # Short: price below 1d EMA20, below 1w EMA20, volume confirm
            short_condition = (price < ema_20_val) and (price < ema_20_1w_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.5x ATR)
            if position == 1:
                if price < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below 1d EMA20)
                elif price < ema_20_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above 1d EMA20)
                elif price > ema_20_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Adaptive_Volume_Trend_Filter_v1"
timeframe = "1d"
leverage = 1.0