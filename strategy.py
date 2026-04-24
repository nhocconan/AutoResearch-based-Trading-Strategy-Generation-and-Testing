#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla pivot breakout with 4h EMA50 trend filter and volume confirmation.
- Long when price breaks above Camarilla H3 level AND 4h close > 4h EMA50 (bullish regime)
- Short when price breaks below Camarilla L3 level AND 4h close < 4h EMA50 (bearish regime)
- Volume confirmation: current volume > 1.5 * 20-period average volume
- Exit on opposite Camarilla breakout (L3 for long exit, H3 for short exit)
- Uses 1h primary with 4h HTF for signal direction to reduce overtrading (target 60-150 trades over 4 years)
- Session filter: only trade 08-20 UTC to avoid low-volume periods
- Discrete position sizing: 0.20 (20% of capital) to manage drawdown
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate typical price for Camarilla (using previous bar's OHLC)
    typical_price = (high + low + close) / 3.0
    # Shift by 1 to use previous bar's typical price (no look-ahead)
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan  # First bar has no previous
    
    # Camarilla levels based on previous bar's range
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    range_prev = prev_high - prev_low
    H3 = prev_typical + range_prev * 1.1 / 4
    L3 = prev_typical - range_prev * 1.1 / 4
    H4 = prev_typical + range_prev * 1.1 / 2
    L4 = prev_typical - range_prev * 1.1 / 2
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (waits for completed 4h bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_4h_aligned
    bearish_regime = close < ema_50_4h_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(H3[i]) or np.isnan(L3[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_confirm[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 AND bullish regime AND volume confirmation
            if close[i] > H3[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below L3 AND bearish regime AND volume confirmation
            elif close[i] < L3[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: break below L3 (opposite Camarilla level)
            if close[i] < L3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: break above H3 (opposite Camarilla level)
            if close[i] > H3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0