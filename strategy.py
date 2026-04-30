#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume confirmation.
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend via smoothed moving averages.
# Long when Alligator is bullish (LIPS > TEETH > JAW) AND price > 1d EMA50 AND volume > 1.8x 20-bar average.
# Short when Alligator is bearish (LIPS < TEETH < JAW) AND price < 1d EMA50 AND volume > 1.8x 20-bar average.
# Exit when Alligator becomes neutral (TEETH crosses JAW or LIPS) or volume drops.
# Uses 1d HTF for trend filter to ensure alignment with higher timeframe direction.
# Discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.
# Works in bull/bear via 1d EMA50 trend filter and volume confirmation to avoid false signals.

name = "12h_Williams_Alligator_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 12h timeframe: SMMA (Smoothed Moving Average)
    # JAW: 13-period SMMA of median price, shifted 8 bars
    # TEETH: 8-period SMMA of median price, shifted 5 bars
    # LIPS: 5-period SMMA of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_PRICE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Shift as per Alligator definition: JAW 8, TEETH 5, LIPS 3
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 13+8, 8+5, 5+3)  # warmup for EMA50, volume MA, and Alligator
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Alligator conditions
        bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: bullish Alligator, uptrend (price > 1d EMA50), volume confirmation
            if bullish and curr_close > ema_50_1d_aligned[i] and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: bearish Alligator, downtrend (price < 1d EMA50), volume confirmation
            elif bearish and curr_close < ema_50_1d_aligned[i] and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit: Alligator turns neutral or bearish, or volume drops
            if not bullish or not curr_volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns neutral or bullish, or volume drops
            if not bearish or not curr_volume_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals