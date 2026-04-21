#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop
Hypothesis: Daily Donchian(20) breakout with 1-week EMA50 trend filter and volume confirmation.
Long when price breaks above 20-day high, above 1w EMA50, and volume > 1.5x 20-day MA.
Short when price breaks below 20-day low, below 1w EMA50, and volume > 1.5x 20-day MA.
Uses ATR-based stop (2.0x) to manage risk. Designed for low trade frequency (~10-20/year)
to work in both bull and bear markets via 1w trend alignment. Avoids overtrading by using
strict breakout conditions and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for EMA trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w EMA50 for trend regime ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Daily ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.5x 20-day MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Daily Donchian channels (20-period) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_1w_val = ema_50_1w_aligned[i]
        vol_avg = vol_ma[i]
        upper_channel = highest_20[i]
        lower_channel = lowest_20[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper channel, above 1w EMA50, volume confirm
            long_condition = (price > upper_channel) and (price > ema_50_1w_val) and volume_confirm
            # Short: price breaks below lower channel, below 1w EMA50, volume confirm
            short_condition = (price < lower_channel) and (price < ema_50_1w_val) and volume_confirm
            
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
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price below 1w EMA50)
                elif price < ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Trend reversal exit (price above 1w EMA50)
                elif price > ema_50_1w_val:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike_ATRStop"
timeframe = "1d"
leverage = 1.0