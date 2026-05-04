#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot R3/S3 Breakout + 1d EMA34 Trend Filter + Volume Spike
# Camarilla pivots identify key intraday support/resistance levels. Breakout above R3 or below S3
# with volume confirmation indicates strong momentum. 1d EMA34 ensures alignment with daily trend
# to avoid counter-trend trades. Designed for 12h timeframe to target 50-150 trades over 4 years
# (12-37/year) minimizing fee drag while capturing strong trending moves. Works in bull markets
# via long breakouts and bear markets via short breakdowns.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels for 12h timeframe using prior 12h bar's OHLC
    # Camarilla levels: based on previous day's range
    # We'll use prior 12h bar's high-low to calculate levels for current 12h bar
    # Shift high/low by 1 to get prior bar's values
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # Set first value to NaN as there's no prior bar
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    # R4 = close + ((high-low) * 1.1/2)
    # R3 = close + ((high-low) * 1.1/4)
    # S3 = close - ((high-low) * 1.1/4)
    # S4 = close - ((high-low) * 1.1/2)
    # We focus on R3 and S3 for breakouts
    hl_range = prev_high - prev_low
    camarilla_r3 = prev_close + (hl_range * 1.1 / 4)
    camarilla_s3 = prev_close - (hl_range * 1.1 / 4)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: close breaks above Camarilla R3 AND 1d uptrend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_aligned[i] and  # 1d uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: close breaks below Camarilla S3 AND 1d downtrend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_aligned[i] and  # 1d downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: close closes below Camarilla R3 OR 1d trend turns down
            if (close[i] < camarilla_r3[i] or 
                close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: close closes above Camarilla S3 OR 1d trend turns up
            if (close[i] > camarilla_s3[i] or 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals