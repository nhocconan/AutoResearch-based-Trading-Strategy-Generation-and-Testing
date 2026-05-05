#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w EMA trend filter + volume confirmation
# Williams %R: momentum oscillator identifying overbought/oversold conditions
# Long when: %R crosses above -80 (oversold bounce) AND 1w EMA(50) > EMA(200) (bullish trend) AND volume > 1.5x 20-period MA
# Short when: %R crosses below -20 (overbought reversal) AND 1w EMA(50) < EMA(200) (bearish trend) AND volume > 1.5x 20-period MA
# Exit when: %R crosses -50 (mean reversion) OR volume < 1.2x 20-period MA (loss of conviction)
# Uses Williams %R for momentum timing, 1w EMA cross for regime filter, volume for conviction
# Timeframe: 1d, HTF: 1w for EMA trend. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_WilliamsR_1wEMA_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14) on 1d
    if len(high) >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero when highest_high == lowest_low
        williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    else:
        williams_r = np.full(n, np.nan)
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_long_signal = (williams_r > -80) & (np.roll(williams_r, 1) <= -80)
    williams_short_signal = (williams_r < -20) & (np.roll(williams_r, 1) >= -20)
    williams_exit_signal = (williams_r > -50) & (np.roll(williams_r, 1) <= -50)  # exit long
    williams_exit_signal_short = (williams_r < -50) & (np.roll(williams_r, 1) >= -50)  # exit short
    
    # Volume confirmation on 1d
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
        volume_weak = volume < (1.2 * vol_ma_20)  # for exit condition
    else:
        volume_filter = np.zeros(n, dtype=bool)
        volume_weak = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA trend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # need sufficient data for EMA
        return np.zeros(n)
    
    # Calculate EMA(50) and EMA(200) on 1w
    close_1w = df_1w['close'].values
    if len(close_1w) >= 200:
        ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_200 = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_bullish = ema_50 > ema_200  # bullish trend
        ema_bearish = ema_50 < ema_200  # bearish trend
    else:
        ema_bullish = np.zeros(len(df_1w), dtype=bool)
        ema_bearish = np.zeros(len(df_1w), dtype=bool)
    
    # Align 1w EMA trend to 1d timeframe
    ema_bullish_aligned = align_htf_to_ltf(prices, df_1w, ema_bullish.astype(float))
    ema_bearish_aligned = align_htf_to_ltf(prices, df_1w, ema_bearish.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(williams_long_signal[i]) or np.isnan(williams_short_signal[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_bullish_aligned[i]) or 
            np.isnan(ema_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R bullish cross + bullish trend + volume filter
            if (williams_long_signal[i] and 
                ema_bullish_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R bearish cross + bearish trend + volume filter
            elif (williams_short_signal[i] and 
                  ema_bearish_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses -50 OR weak volume
            if (williams_exit_signal[i] or volume_weak[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses -50 OR weak volume
            if (williams_exit_signal_short[i] or volume_weak[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals