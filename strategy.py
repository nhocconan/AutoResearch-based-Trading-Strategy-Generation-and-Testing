#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2
Hypothesis: Camarilla pivot breakouts at R1/S1 on 12h filtered by 1d EMA34 trend and volume spike (>2.0x 20-period average).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn.
1d trend filter provides robust directional bias across bull/bear markets while reducing whipsaws.
Target: 12-37 trades/year per symbol for low fee drag and strong test generalization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA34 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h OHLC for Camarilla calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate previous day's Camarilla levels (using prior 1d bar's daily range)
    # We need daily high/low from 1d data to compute Camarilla for current 12h period
    cam_high = df_1d['high'].values
    cam_low = df_1d['low'].values
    cam_close = df_1d['close'].values
    
    # Camarilla levels: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low),
    # R2 = close + 0.55*(high-low), R1 = close + 0.275*(high-low)
    # S1 = close - 0.275*(high-low), S2 = close - 0.55*(high-low),
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    rng = cam_high - cam_low
    r1 = cam_close + 0.275 * rng
    s1 = cam_close - 0.275 * rng
    r4 = cam_close + 1.5 * rng
    s4 = cam_close - 1.5 * rng
    
    # Align Camarilla levels to 12h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 1d EMA34 for trend filter ===
    ema_34_1d = pd.Series(cam_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) 
            or np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 2.0x 20-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
            vol_filter = volume[i] > 2.0 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > R1, 1d uptrend, volume filter
            long_breakout = price > r1_aligned[i]
            long_trend = price > ema_34_1d_aligned[i]
            
            # Short conditions: price < S1, 1d downtrend, volume filter
            short_breakout = price < s1_aligned[i]
            short_trend = price < ema_34_1d_aligned[i]
            
            # Entry logic - ONLY enter on volume filter + trend alignment
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below S1 (breakdown)
            elif price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above R1 (breakout)
            elif price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ATRStop_v2"
timeframe = "12h"
leverage = 1.0