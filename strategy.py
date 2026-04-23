#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA34 trend filter and volume spike.
- Uses 4h HTF for trend alignment (stable, less noisy than 1h)
- Camarilla H3/L3 from prior 4h for structure (tighter levels than H4/L4)
- Long: price breaks above H3 + volume > 1.8x 20-period avg + price > 4h EMA34
- Short: price breaks below L3 + volume > 1.8x 20-period avg + price < 4h EMA34
- Exit: price re-enters Camarilla H3-L3 range OR 4h EMA34 trend flip
- Discrete position sizing: ±0.20 to minimize fee churn
- Session filter: 08-20 UTC to avoid low-volume hours
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.8x 20-period average (tight to avoid overtrading)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h EMA34 for trend filter (HTF = 4h as specified)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Calculate 4h Camarilla levels (based on prior 4h OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_prev = df_4h['close'].values
    
    # Camarilla formula: range = high - low
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    rng = high_4h - low_4h
    camarilla_h3 = close_4h_prev + rng * (1.1 / 4)
    camarilla_l3 = close_4h_prev - rng * (1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Session filter: 08-20 UTC (already datetime64[ms] index)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 34 for EMA34, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0 and in_session:
            # Long: price breaks above H3 + volume confirmation + price > 4h EMA34
            if (close[i] > h3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3 + volume confirmation + price < 4h EMA34
            elif (close[i] < l3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_34_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price re-enters below L3 (mean reversion) OR price < 4h EMA34 (trend flip)
            if close[i] < l3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price re-enters above H3 (mean reversion) OR price > 4h EMA34 (trend flip)
            if close[i] > h3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0