#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Long when Alligator jaws (13-period SMMA) > teeth (8-period SMMA) > lips (5-period SMMA) AND close > 1d EMA50 AND volume > 1.5x 20-bar avg
# Short when Alligator jaws < teeth < lips AND close < 1d EMA50 AND volume > 1.5x 20-bar avg
# Uses Williams Alligator (smoothed moving averages) to identify trending markets, EMA50 for higher timeframe trend filter
# Volume confirmation ensures breakouts have conviction. Discrete position sizing (0.25) minimizes fee churn.
# Target: 20-50 trades/year on 4h. Works in bull markets by capturing trends with Alligator alignment, works in bear by requiring volume spikes
# which often accompany exhaustion moves preceding reversals.

name = "4h_WilliamsAlligator_1dEMA50_Trend_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is simple SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Williams Alligator components (5, 8, 13 period SMMA)
    # Note: Alligator uses SMMA with specific shifts
    lips = smma(close, 5)      # 5-period SMMA
    teeth = smma(close, 8)     # 8-period SMMA
    jaws = smma(close, 13)     # 13-period SMMA
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: >1.5x 20-bar average volume (moderate filter to balance signal quality and frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaws[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_1d_aligned[i]
        lip_val = lips[i]
        tooth_val = teeth[i]
        jaw_val = jaws[i]
        curr_close = close[i]
        
        # Alligator alignment: lips > teeth > jaws (bullish) or lips < teeth < jaws (bearish)
        bullish_alignment = lip_val > tooth_val > jaw_val
        bearish_alignment = lip_val < tooth_val < jaw_val
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Alligator bullish AND close > 1d EMA50 AND volume confirmation
            if bullish_alignment and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator bearish AND close < 1d EMA50 AND volume confirmation
            elif bearish_alignment and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Alligator loses bullish alignment
            if not bullish_alignment:  # Lips <= teeth or teeth <= jaws
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Alligator loses bearish alignment
            if not bearish_alignment:  # Lips >= teeth or teeth >= jaws
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals