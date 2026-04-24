#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Long when price breaks above Camarilla H3 AND 4h close > 4h EMA50 (bullish regime)
- Short when price breaks below Camarilla L3 AND 4h close < 4h EMA50 (bearish regime)
- Volume confirmation: current volume > 2.0 * 20-period average volume (strong spike)
- Exit on opposite Camarilla breakout (L3 for long exit, H3 for short exit)
- Uses 1h primary with 4h HTF to target 60-150 trades over 4 years (15-37/year)
- Camarilla levels provide precise intraday support/resistance; EMA50 filters regime; volume spike confirms momentum
- Designed to work in both bull (breakouts) and bear (mean reversion at extremes) markets
- Signal size: 0.20 discrete levels to minimize fee churn
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
    
    # Calculate Camarilla levels using previous day's OHLC (avoid look-ahead)
    # We need daily OHLC, so we'll use 1d data shifted appropriately
    # For 1h bars, we use the previous completed 1d bar's OHLC
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Get daily OHLC arrays
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each daily bar
    # H3 = daily_close + (daily_high - daily_low) * 1.1/4
    # L3 = daily_close - (daily_high - daily_low) * 1.1/4
    daily_range = daily_high - daily_low
    camarilla_h3 = daily_close + daily_range * 1.1 / 4
    camarilla_l3 = daily_close - daily_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (waits for completed 1d bar)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Trend filter: bullish if close > EMA50, bearish if close < EMA50
    bullish_regime = close > ema_50_4h_aligned
    bearish_regime = close < ema_50_4h_aligned
    
    # Volume confirmation: volume > 2.0 * 20-period average (strong spike)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need EMA50 and Camarilla (based on 1d data)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Camarilla H3 AND bullish regime AND volume confirmation
            if close[i] > camarilla_h3_aligned[i] and bullish_regime[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below Camarilla L3 AND bearish regime AND volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and bearish_regime[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: break below Camarilla L3 (opposite level)
            if close[i] < camarilla_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: break above Camarilla H3 (opposite level)
            if close[i] > camarilla_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0