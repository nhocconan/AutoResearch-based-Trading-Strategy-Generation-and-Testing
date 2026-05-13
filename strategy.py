#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND close > 1w EMA50 AND volume > 1.5x average
# Short when price breaks below Camarilla S3 AND close < 1w EMA50 AND volume > 1.5x average
# Exit when price crosses Camarilla pivot point (mean reversion) OR trend reversal (price crosses 1w EMA50)
# Uses 12h timeframe for lower trade frequency (target: 50-150 trades over 4 years), Camarilla for structure,
# 1w EMA for strong trend filter, volume spike for confirmation. Works in bull via breakout continuation,
# bear via faded rallies and mean reversion to pivot.

name = "12h_Camarilla_R3S3_Breakout_1wEMA50_Volume_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 12h data for Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels on 12h data (using previous day's range)
    # Need previous day's high/low - we'll use rolling window of 2 bars (approx 1 day at 12h)
    if len(high_12h) >= 2:
        prev_day_high = pd.Series(high_12h).rolling(window=2, min_periods=2).max().shift(1).values
        prev_day_low = pd.Series(low_12h).rolling(window=2, min_periods=2).min().shift(1).values
        prev_day_close = pd.Series(close_12h).rolling(window=2, min_periods=2).last().shift(1).values
    else:
        prev_day_high = np.full_like(high_12h, np.nan)
        prev_day_low = np.full_like(low_12h, np.nan)
        prev_day_close = np.full_like(close_12h, np.nan)
    
    # Camarilla formulas
    R3 = prev_day_close + (prev_day_high - prev_day_low) * 1.1 / 4
    S3 = prev_day_close - (prev_day_high - prev_day_low) * 1.1 / 4
    PP = (prev_day_high + prev_day_low + prev_day_close) / 3  # Pivot Point
    
    # Align Camarilla levels to 12h timeframe (already aligned since calculated on 12h)
    # But we need to ensure proper alignment with look-ahead prevention
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    PP_aligned = align_htf_to_ltf(prices, df_12h, PP)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current 12h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_12h > (1.5 * vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for Camarilla and EMA
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(PP_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > R3 AND close > 1w EMA50 AND volume spike
            if close[i] > R3_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < S3 AND close < 1w EMA50 AND volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < PP (mean reversion) OR trend reversal (close < 1w EMA50)
            if close[i] < PP_aligned[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > PP (mean reversion) OR trend reversal (close > 1w EMA50)
            if close[i] > PP_aligned[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals