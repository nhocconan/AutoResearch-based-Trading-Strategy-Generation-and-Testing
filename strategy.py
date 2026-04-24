#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla H3 AND 1w close > 1w EMA50 (bullish regime)
- Short when price breaks below Camarilla L3 AND 1w close < 1w EMA50 (bearish regime)
- Volume confirmation: current volume > 2.0 * 20-period average volume (strong spike)
- Exit on opposite Camarilla breakout (L3 for long exit, H3 for short exit)
- Uses 12h primary with 1w HTF to target 50-150 trades over 4 years (12-37/year)
- Camarilla levels provide precise support/resistance; EMA50 filters regime; volume spike confirms momentum
- Designed to work in both bull (breakouts) and bear (mean reversion at extremes) markets
- Signal size: 0.25 discrete levels to minimize fee churn
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
    
    # Calculate Camarilla levels using previous week's OHLC (avoid look-ahead)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get weekly OHLC arrays
    weekly_open = df_1w['open'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each weekly bar
    # H3 = weekly_close + (weekly_high - weekly_low) * 1.1/4
    # L3 = weekly_close - (weekly_high - weekly_low) * 1.1/4
    weekly_range = weekly_high - weekly_low
    camarilla_h3 = weekly_close + weekly_range * 1.1 / 4
    camarilla_l3 = weekly_close - weekly_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (waits for completed 1w bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_1w_aligned
    bearish_regime = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need EMA50 and Camarilla (based on 1w data)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 AND bullish regime AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla L3 AND bearish regime AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Camarilla L3 (opposite level)
            if close[i] < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Camarilla H3 (opposite level)
            if close[i] > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0