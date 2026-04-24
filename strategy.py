#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h for execution, HTF: 4h for EMA trend.
- Camarilla levels calculated from previous 4h bar (H3/L3 = close ± 1.1*(high-low)/6).
- Entry: Long when price breaks above H3 with volume spike and price > 4h EMA50 (uptrend).
         Short when price breaks below L3 with volume spike and price < 4h EMA50 (downtrend).
- Exit: When price returns to the 4h VWAP (mean reversion to fair value).
- Works in bull via buying strength above resistance, in bear via selling weakness below support.
- Discrete signal size: 0.20 to limit drawdown and reduce fee churn.
- Session filter: 08-20 UTC to avoid low-volume noise.
- Target: 60-150 total trades over 4 years = 15-37/year for 1h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50 = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h VWAP for exit (mean reversion target)
    typical_price = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    vwap = (typical_price * df_4h['volume']).cumsum() / df_4h['volume'].cumsum()
    vwap_values = vwap.values
    
    # Calculate Camarilla levels from previous 4h bar
    # H3 = close + 1.1*(high-low)/6, L3 = close - 1.1*(high-low)/6
    camarilla_high = df_4h['close'] + 1.1 * (df_4h['high'] - df_4h['low']) / 6
    camarilla_low = df_4h['close'] - 1.1 * (df_4h['high'] - df_4h['low']) / 6
    
    # Align 4h indicators to 1h
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50)
    vwap_aligned = align_htf_to_ltf(prices, df_4h, vwap_values)
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high.values)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA (on 1h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 + volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(volume_spike[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for breakout signals with volume spike and trend filter
            if volume_spike[i]:
                # Long: price breaks above H3 with uptrend
                if close[i] > camarilla_high_aligned[i] and close[i] > ema_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below L3 with downtrend
                elif close[i] < camarilla_low_aligned[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price returns to VWAP (mean reversion)
            if close[i] <= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to VWAP (mean reversion)
            if close[i] >= vwap_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VWAP_MeanReversion_v1"
timeframe = "1h"
leverage = 1.0