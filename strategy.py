#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA50 trend direction.
- EMA50 > rising: bullish bias, EMA50 < falling: bearish bias.
- Entry: Long when price breaks above Camarilla H3 AND 4h EMA50 trending up.
         Short when price breaks below Camarilla L3 AND 4h EMA50 trending down.
- Exit: Opposite Camarilla break (L3 for long, H3 for short) or EMA trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Session filter: 08-20 UTC to reduce noise trades.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h close
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate Camarilla levels (H3, L3) on 1h using previous bar's range
    # Camarilla: H3 = close + 1.1 * (high - low) / 2, L3 = close - 1.1 * (high - low) / 2
    # Use previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # avoid NaN on first bar
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    h3 = prev_close + 1.1 * camarilla_range / 2.0
    l3 = prev_close - 1.1 * camarilla_range / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(volume_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        ema_50_prev = ema_50_aligned[i-1] if i > 0 else ema_50_val
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Determine 4h EMA50 trend: rising if current > previous
                ema50_rising = ema_50_val > ema_50_prev
                ema50_falling = ema_50_val < ema_50_prev
                
                # Bullish breakout: price breaks above H3 AND EMA50 rising
                if curr_high > h3[i] and ema50_rising:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below L3 AND EMA50 falling
                elif curr_low < l3[i] and ema50_falling:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA50 starts falling
            if curr_low < l3[i] or ema_50_val < ema_50_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA50 starts rising
            if curr_high > h3[i] or ema_50_val > ema_50_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0