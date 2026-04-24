#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 Breakout with 4h EMA34 Trend Filter and Volume Spike.
- Uses 4h EMA34 for higher-timeframe trend alignment (bullish/bearish filter).
- Camarilla H3/L3 from 1d chart as key support/resistance for breakouts.
- Volume spike (>2x 24-period average on 1h) confirms breakout validity.
- Session filter: only trade 08-20 UTC to avoid low-liquidity hours.
- Discrete position sizing (0.20) to minimize fee churn.
- Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag on 1h.
- Works in bull/bear via 4h trend filter and volatility-based volume confirmation.
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
    
    # Precompute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data ONCE for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla H3 and L3 from 1d OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 1h timeframe (wait for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Get 4h data ONCE for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: > 2.0x 24-period average volume (24 * 1h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above H3 with volume spike and above 4h EMA34 (bullish higher-timeframe trend)
            if close[i] > camarilla_h3_aligned[i] and volume_spike[i] and close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below L3 with volume spike and below 4h EMA34 (bearish higher-timeframe trend)
            elif close[i] < camarilla_l3_aligned[i] and volume_spike[i] and close[i] < ema_34_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR below 4h EMA34 (trend change)
            if close[i] < camarilla_l3_aligned[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price closes above H3 OR above 4h EMA34 (trend change)
            if close[i] > camarilla_h3_aligned[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA34_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0