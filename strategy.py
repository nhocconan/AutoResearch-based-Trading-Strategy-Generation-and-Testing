#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1w trend filter and volume confirmation.
- Primary timeframe: 6h for entries.
- HTF: 1w Camarilla pivot levels (H4/L4 = strong resistance/support) for structure.
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 6h volume > 1.5 * 20-period 6h volume MA to confirm breakout.
- Entry: Long when price breaks above H4 AND 1w trend bullish AND volume spike.
         Short when price breaks below L4 AND 1w trend bearish AND volume spike.
- Exit: Opposite breakout (price re-enters pivot range) or loss of volume confirmation.
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Camarilla pivots work well in ranging markets (common in 2025 bear) and capture breakouts in trending markets.
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
    
    # Get 1w data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w Camarilla pivots (based on previous week's OHLC)
    # Camarilla levels: H4 = close + 1.5*(high-low), H3 = close + 1.125*(high-low), etc.
    # We use H4/L4 as strong breakout levels
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Previous week's range
    prev_high = np.roll(df_1w_high, 1)
    prev_low = np.roll(df_1w_low, 1)
    prev_close = np.roll(df_1w_close, 1)
    # First week has no previous data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla H4 and L4
    camarilla_h4 = prev_close + 1.5 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.5 * (prev_high - prev_low)
    
    # Calculate 1w EMA50 for trend
    ema50_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish = df_1w_close > ema50_1w  # True if bullish
    trend_bearish = df_1w_close < ema50_1w  # True if bearish
    
    # Align HTF indicators to 6h
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish.astype(float))
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA
    vol_ma_6h = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need enough bars for 1w EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        bullish_trend = trend_bullish_aligned[i] > 0.5
        bearish_trend = trend_bearish_aligned[i] > 0.5
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price above H4 AND 1w trend bullish
                if curr_low > h4_level and bullish_trend:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price below L4 AND 1w trend bearish
                elif curr_high < l4_level and bearish_trend:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters pivot range (below H4) OR loss of volume confirmation
            if curr_high < h4_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters pivot range (above L4) OR loss of volume confirmation
            if curr_low > l4_level or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_1wTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0