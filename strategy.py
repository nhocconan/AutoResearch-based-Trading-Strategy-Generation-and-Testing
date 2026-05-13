#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 with 4h EMA34 uptrend and volume > 2.0x average.
# Short when price breaks below Camarilla S3 with 4h EMA34 downtrend and volume > 2.0x average.
# Uses ATR(14) trailing stop (2.5x) for risk control. Discrete sizing 0.20.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h.
# Session filter: 08-20 UTC to reduce noise trades. Uses 4h/1d for signal direction, 1h only for entry timing.

name = "1h_Camarilla_R3S3_Breakout_4hEMA34_VolumeSpike_ATRStop_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for trailing stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    session_mask = (prices.index.hour >= 8) & (prices.index.hour <= 20)
    
    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels using previous 1d bar
    # Camarilla R3 = close + (high - low) * 1.1/4
    # Camarilla S3 = close - (high - low) * 1.1/4
    camarilla_range = high_1d - low_1d
    camarilla_r3 = close_1d + camarilla_range * 1.1 / 4
    camarilla_s3 = close_1d - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe (wait for daily bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA34
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 4h EMA34 to 1h timeframe (wait for 4h bar to close)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade between 08-20 UTC
        if not session_mask[i]:
            signals[i] = 0.0
            # Carry forward tracking values when flat
            if position == 0 and i > 0:
                highest_since_entry[i] = highest_since_entry[i-1]
                lowest_since_entry[i] = lowest_since_entry[i-1]
            elif position == 1 and i > 0:
                highest_since_entry[i] = highest_since_entry[i-1]
            elif position == -1 and i > 0:
                lowest_since_entry[i] = lowest_since_entry[i-1]
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 AND 4h EMA34 uptrend AND volume > 2.0x average
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema34_4h_aligned[i] and  # Price above 4h EMA34 confirms uptrend
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price breaks below Camarilla S3 AND 4h EMA34 downtrend AND volume > 2.0x average
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema34_4h_aligned[i] and  # Price below 4h EMA34 confirms downtrend
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
                lowest_since_entry[i] = low[i]  # Initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward tracking values when flat
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
                    lowest_since_entry[i] = lowest_since_entry[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_since_entry[i] = max(highest_since_entry[i-1], high[i])
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                highest_since_entry[i] = np.nan
            else:
                signals[i] = 0.20
                # Carry forward tracking
                if i > 0:
                    highest_since_entry[i] = highest_since_entry[i-1]
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry[i] = min(lowest_since_entry[i-1], low[i])
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
            if trailing_stop:
                signals[i] = 0.0
                position = 0
                # Reset tracking when flat
                lowest_since_entry[i] = np.nan
            else:
                signals[i] = -0.20
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals