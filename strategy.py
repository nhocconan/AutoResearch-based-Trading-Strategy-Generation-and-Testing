#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h for entries/exits.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 4h volume > 2.0 * 20-period volume MA to avoid false breakouts.
- Entry: Long when price breaks above R1 AND 12h EMA50 trend bullish AND volume spike.
         Short when price breaks below S1 AND 12h EMA50 trend bearish AND volume spike.
- Exit: Opposite Camarilla level touch (S1 for long, R1 for short) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
Camarilla pivot levels provide precise intraday support/resistance that work in both bull and bear markets.
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
    
    # Calculate prior day's OHLC for Camarilla levels
    # Shift by 96 bars (4h * 6 = 24h) to get prior day's values
    prior_close = np.roll(close, 96)
    prior_high = np.roll(high, 96)
    prior_low = np.roll(low, 96)
    # Set first 96 values to NaN since they don't have prior day data
    prior_close[:96] = np.nan
    prior_high[:96] = np.nan
    prior_low[:96] = np.nan
    
    # Calculate Camarilla levels for current day
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rng = prior_high - prior_low
    r1 = prior_close + rng * 1.1 / 12
    s1 = prior_close - rng * 1.1 / 12
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    df_12h_close = df_12h['close'].values
    ema_50 = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period volume MA on 12h for volume confirmation
    df_12h_volume = df_12h['volume'].values
    vol_ma_20 = pd.Series(df_12h_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    # Volume confirmation: current 4h volume > 2.0 * 20-period 12h volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_20_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(96, 50, 20)  # Need enough bars for Camarilla and 12h indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_50_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: price breaks above R1 AND 12h EMA50 trend bullish
                if curr_high > r1[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below S1 AND 12h EMA50 trend bearish
                elif curr_low < s1[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price touches S1 OR loss of volume confirmation
            if curr_low <= s1[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price touches R1 OR loss of volume confirmation
            if curr_high >= r1[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0