#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_HTFTrend_VolumeSpike_ATRStop_v2
Hypothesis: Camarilla pivot breakouts at R1/S1 on 4h filtered by 1d EMA50 trend and volume spike (>1.8x 30-period average).
Uses ATR(14) stoploss (2.0x) and discrete position sizing (0.25) to minimize fee churn.
1d trend filter provides robust directional bias across bull/bear markets while reducing whipsaws.
Target: 19-50 trades/year per symbol for low fee drag and strong test generalization.
Improved from v1: using 1d instead of 12h for HTF trend, slightly looser volume filter to increase trade frequency while maintaining edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50 trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h OHLC for Camarilla calculation ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate previous day's Camarilla levels (using prior 1d bar's daily range)
    cam_high = df_1d['high'].values
    cam_low = df_1d['low'].values
    cam_close = df_1d['close'].values
    
    # Camarilla levels: R1 = close + 0.275*(high-low), S1 = close - 0.275*(high-low)
    rng = cam_high - cam_low
    r1 = cam_close + 0.275 * rng
    s1 = cam_close - 0.275 * rng
    
    # Align Camarilla levels to 4h timeframe (use prior day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(cam_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
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
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Volume filter: current volume > 1.8x 30-period average
            volume = prices['volume'].values
            vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
            vol_filter = volume[i] > 1.8 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
            
            # Long conditions: price > R1, 1d uptrend, volume filter
            long_breakout = price > r1_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < S1, 1d downtrend, volume filter
            short_breakout = price < s1_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            
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

name = "4h_Camarilla_R1_S1_Breakout_HTFTrend_VolumeSpike_ATRStop_v2"
timeframe = "4h"
leverage = 1.0