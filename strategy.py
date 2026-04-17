#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation
- Uses 4h EMA50 slope for trend bias (long when rising, short when falling)
- Breakout triggers when price closes beyond H3 (long) or L3 (short) with volume > 2.0x 20-period MA
- Fixed position size 0.20 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.0x ATR) to lock in profits and reduce losses
- Session filter: only trade between 08:00-20:00 UTC to avoid low-liquidity hours
- Works in bull markets (buying H3 breakouts in uptrends) and bear markets (selling L3 breakdowns in downtrends)
- Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute hour for session filter (08:00-20:00 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA50 trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 and its slope
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_slope = np.gradient(ema50_4h)  # slope of EMA50
    
    # Get 1d data for Camarilla pivot levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (H3, L3) from previous completed 1d bar
    rng = high_1d - low_1d
    h3 = close_1d + 1.1 * rng / 4
    l3 = close_1d - 1.1 * rng / 4
    # Shift by 1 to use only completed 1d bars (avoid look-ahead)
    h3_prev = np.roll(h3, 1)
    l3_prev = np.roll(l3, 1)
    h3_prev[0] = h3[0]
    l3_prev[0] = l3[0]
    
    # Align all indicators to 1h timeframe (primary)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_prev)
    ema50_slope_aligned = align_htf_to_ltf(prices, df_4h, ema50_slope)
    
    # Volume average (20-period) on 1h
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 1h for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Session filter: only trade 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema50_slope_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema_slope = ema50_slope_aligned[i]
        vol_ma = volume_ma_20[i]
        atr_val = atr_14[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price closes above H3 + volume spike + EMA50 rising
            if price > h3_val and vol > 2.0 * vol_ma and ema_slope > 0:
                signals[i] = 0.20
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below L3 + volume spike + EMA50 falling
            elif price < l3_val and vol > 2.0 * vol_ma and ema_slope < 0:
                signals[i] = -0.20
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeSpike_ATRTrail"
timeframe = "1h"
leverage = 1.0