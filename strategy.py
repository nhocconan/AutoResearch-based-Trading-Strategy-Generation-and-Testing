#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d EMA34 trend filter + volume confirmation.
# Alligator: Jaw (13-period SMMA, offset 8), Teeth (8-period SMMA, offset 5), Lips (5-period SMMA, offset 3).
# Long when Lips > Teeth > Jaw (bullish alignment) AND price > 1d EMA34 AND volume > 2.0x 20-bar average.
# Short when Lips < Teeth < Jaw (bearish alignment) AND price < 1d EMA34 AND volume > 2.0x 20-bar average.
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.
# Works in bull/bear via 1d EMA34 trend filter and Alligator alignment.

name = "6h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator components (SMMA with offsets)
    def smma(values, period):
        """Smoothed Moving Average"""
        result = np.full_like(values, np.nan, dtype=np.float64)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Jaw: 13-period SMMA, offset 8 bars
    jaw_raw = smma(close, 13)
    jaw = np.roll(jaw_raw, 8)  # shift right by 8 (offset into future)
    # Teeth: 8-period SMMA, offset 5 bars
    teeth_raw = smma(close, 8)
    teeth = np.roll(teeth_raw, 5)  # shift right by 5
    # Lips: 5-period SMMA, offset 3 bars
    lips_raw = smma(close, 5)
    lips = np.roll(lips_raw, 3)  # shift right by 3
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for EMA34 and Alligator
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        is_uptrend = close[i] > ema_34_1d_aligned[i]
        is_downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Alligator alignment
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator + uptrend + volume confirmation
            if bullish_alignment and is_uptrend and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator + downtrend + volume confirmation
            elif bearish_alignment and is_downtrend and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when bullish alignment breaks or trend reverses
            if not bullish_alignment or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bearish alignment breaks or trend reverses
            if not bearish_alignment or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals