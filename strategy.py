#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Long when Alligator jaws (blue line) turns up AND price > Alligator teeth (red line) AND price > 1d EMA34 AND volume > 1.5x 24-bar avg
# Short when Alligator jaws turn down AND price < Alligator teeth AND price < 1d EMA34 AND volume > 1.5x 24-bar avg
# Exit when price crosses Alligator teeth in opposite direction
# Uses 12h timeframe to reduce trade frequency (target: 12-30 trades/year) and avoid fee drag
# Williams Alligator (SMMA(13,8), SMMA(8,5), SMMA(5,3)) identifies trend formation and direction
# 1d EMA34 filter ensures alignment with higher timeframe trend
# Volume confirmation adds conviction to signals
# Designed to work in both bull and bear markets by following the higher timeframe trend

name = "12h_WilliamsAlligator_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(34) on 1d data
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h timeframe
    # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        result = np.full_like(data, np.nan)
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line (not used in signals but confirms trend)
    
    # Volume confirmation: >1.5x 24-bar average volume (24 * 12h = 12 days lookback)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > 1.5 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 24)  # Need enough data for Alligator Jaw and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        jaw_val = jaw[i]
        teeth_val = teeth[i]
        ema_34 = ema_34_1d_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Alligator jaws turn up (jaw > teeth) AND price > teeth AND price > 1d EMA34 AND volume confirmation
            if jaw_val > teeth_val and curr_close > teeth_val and curr_close > ema_34 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Alligator jaws turn down (jaw < teeth) AND price < teeth AND price < 1d EMA34 AND volume confirmation
            elif jaw_val < teeth_val and curr_close < teeth_val and curr_close < ema_34 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when price crosses below teeth (trend change)
            if curr_close < teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when price crosses above teeth (trend change)
            if curr_close > teeth_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals