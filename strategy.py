#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h for balance of trade frequency and signal quality.
- HTF: 4h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 1h volume > 2.0 * 20-period 4h volume MA to capture institutional interest.
- Camarilla: H3 and L3 levels calculated from prior 4h bar's range.
- Entry: Long when price breaks above H3 AND 4h EMA50 bullish AND volume spike.
         Short when price breaks below L3 AND 4h EMA50 bearish AND volume spike.
- Exit: Price reverts to prior 4h close (pivot) or loss of volume confirmation.
- Signal size: 0.20 discrete to minimize fee churn and manage drawdown.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
This strategy combines volume confirmation with Camarilla pivot breakouts,
filtered by 4h trend to avoid counter-trend trades. Works in both bull and bear markets
by only taking trades in the direction of the 4h trend, with volume spikes confirming
participation. Camarilla levels provide natural support/resistance for mean reversion exits.
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
    
    # Get 4h data for Camarilla levels and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 4h volume MA
    df_4h_volume = df_4h['volume'].values
    vol_ma_4h = pd.Series(df_4h_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Camarilla levels from prior 4h bar's OHLC
    # H3 = close + 1.1 * (high - low) / 4
    # L3 = close - 1.1 * (high - low) / 4
    # Pivot (close) = (high + low + close) / 3
    h4h = df_4h['high'].values
    l4h = df_4h['low'].values
    c4h = df_4h['close'].values
    
    camarilla_h3 = c4h + 1.1 * (h4h - l4h) / 4
    camarilla_l3 = c4h - 1.1 * (h4h - l4h) / 4
    camarilla_pivot = (h4h + l4h + c4h) / 3  # Typical price as pivot
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pivot)
    
    # Volume confirmation: current 1h volume > 2.0 * 20-period 4h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_4h_aligned)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_pivot_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for breakout signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price > H3 AND 4h EMA50 bullish (close > EMA)
                if curr_close > camarilla_h3_aligned[i] and curr_close > ema_4h_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                # Bearish breakout: price < L3 AND 4h EMA50 bearish (close < EMA)
                elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_4h_aligned[i]:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price reverts to pivot OR loss of volume confirmation OR outside session
            if curr_close <= camarilla_pivot_aligned[i] or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price reverts to pivot OR loss of volume confirmation OR outside session
            if curr_close >= camarilla_pivot_aligned[i] or not volume_spike[i] or not in_session[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0