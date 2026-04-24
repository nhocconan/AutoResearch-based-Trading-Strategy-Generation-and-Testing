#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and 1d volume spike confirmation.
- Primary timeframe: 1h for entry/exit timing.
- HTF: 4h EMA50 for trend direction (bullish if price > EMA50, bearish if price < EMA50).
- Volume: Current 1h volume > 2.0 * 20-period 1d volume MA to avoid low-volume breakouts.
- Entry: Long when price breaks above H3 level AND 4h EMA50 bullish AND volume spike.
         Short when price breaks below L3 level AND 4h EMA50 bearish AND volume spike.
- Exit: Opposite Camarilla level (L3 for long, H3 for short) or loss of volume confirmation.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.
Camarilla pivot levels provide high-probability intraday support/resistance. Combined with trend
and volume filters, this avoids false breakouts and works in both bull and bear markets by
only taking trades in the direction of the 4h trend.
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
    
    # Calculate Camarilla pivot levels for 1h (based on previous bar's OHLC)
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # We focus on H3 and L3 for breakouts
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # Using previous bar's OHLC to avoid look-ahead
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    # Set first bar to NaN since no previous bar
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period volume MA on 1d
    df_1d_volume = df_1d['volume'].values
    vol_ma_1d = pd.Series(df_1d_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Volume confirmation: current 1h volume > 2.0 * 20-period 1d volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(volume_spike[i]) or
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above H3 AND 4h EMA50 bullish (price > EMA)
                if curr_high > camarilla_h3[i] and curr_close > ema_val:
                    signals[i] = 0.20
                    position = 1
                # Bearish: price breaks below L3 AND 4h EMA50 bearish (price < EMA)
                elif curr_low < camarilla_l3[i] and curr_close < ema_val:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR loss of volume confirmation
            if curr_low < camarilla_l3[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above H3 OR loss of volume confirmation
            if curr_high > camarilla_h3[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_1dVolumeSpike_v1"
timeframe = "1h"
leverage = 1.0