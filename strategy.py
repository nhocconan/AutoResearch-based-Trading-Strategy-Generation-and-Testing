#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA50 trend direction.
- EMA50 > rising (current > previous) = bullish trend bias, EMA50 < falling = bearish trend bias.
- Entry: Long when price breaks above Camarilla H3 level AND EMA50 bullish bias AND volume spike.
         Short when price breaks below Camarilla L3 level AND EMA50 bearish bias AND volume spike.
- Exit: Opposite Camarilla break (L3 for long, H3 for short) or EMA50 trend bias flip.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to filter weak breakouts).
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Session filter: 08-20 UTC to avoid low-volume Asian session noise.
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
    
    # Precompute session hours filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 4h
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    
    # Calculate EMA50 bias: rising (bullish) if current > previous, falling (bearish) if current < previous
    ema_50_prev = np.roll(ema_50_aligned, 1)
    ema_50_prev[0] = ema_50_aligned[0]  # first bar: no previous
    ema_bias = np.where(ema_50_aligned > ema_50_prev, 1,  # bullish
                        np.where(ema_50_aligned < ema_50_prev, -1, 0))  # bearish/flat
    
    # Camarilla levels (based on previous day's OHLC) - need 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # H3 = close + 1.1*(high-low)/4
    # L3 = close - 1.1*(high-low)/4
    prev_1d_close = df_1d['close'].shift(1).values
    prev_1d_high = df_1d['high'].shift(1).values
    prev_1d_low = df_1d['low'].shift(1).values
    
    camarilla_h3 = prev_1d_close + 1.1 * (prev_1d_high - prev_1d_low) / 4
    camarilla_l3 = prev_1d_close - 1.1 * (prev_1d_high - prev_1d_low) / 4
    
    # Align Camarilla levels to 1h
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 1h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 2)  # Need EMA50 warmup and 1d shift
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h3 = h3_aligned[i]
        l3 = l3_aligned[i]
        bias = ema_bias[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation and trend bias
            if vol_spike:
                # Bullish breakout: price breaks above H3 with bullish EMA bias
                if curr_close > h3 and bias == 1:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price breaks below L3 with bearish EMA bias
                elif curr_close < l3 and bias == -1:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA bias turns bearish
            if curr_close < l3 or bias == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA bias turns bullish
            if curr_close > h3 or bias == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50Trend_VolumeConfirm_v1"
timeframe = "1h"
leverage = 1.0