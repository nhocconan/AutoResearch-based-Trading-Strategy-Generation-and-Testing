#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 1d for low-frequency, high-conviction signals.
- HTF: 1w EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Volume: Current 1d volume > 2.0 * 20-period volume MA to capture institutional interest.
- Entry: Long when price breaks above H3 AND 1w EMA34 bullish AND volume spike.
         Short when price breaks below L3 AND 1w EMA34 bearish AND volume spike.
- Exit: Opposite Camarilla level (L3 for long, H3 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
This strategy targets strong intraday reversals at key Camarilla levels, filtered by weekly trend and volume confirmation,
designed to work in both bull and bear markets by only taking trades in the direction of the 1w trend.
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
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # H3 = close + 1.1 * (high - low) / 2
    # L3 = close - 1.1 * (high - low) / 2
    # Using previous bar's high/low/close to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]  # fill first value
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 20-period 1w volume MA
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 1d volume > 2.0 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need enough bars for EMA34, volume MA, and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(camarilla_h3[i]) or np.isnan(camarilla_l3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_1w_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above H3 AND 1w EMA34 bullish (close > EMA)
                if curr_high > camarilla_h3[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below L3 AND 1w EMA34 bearish (close < EMA)
                elif curr_low < camarilla_l3[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below L3 OR loss of volume confirmation
            if curr_low < camarilla_l3[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR loss of volume confirmation
            if curr_high > camarilla_h3[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0