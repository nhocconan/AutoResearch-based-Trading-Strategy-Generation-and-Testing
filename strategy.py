#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA20 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND price > 4h EMA20 AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 AND price < 4h EMA20 AND volume > 1.5x 20-period average.
# Exit on ATR(14) trailing stop (2.0x). Uses 1h primary timeframe and 4h HTF for trend alignment.
# Camarilla levels provide precise intraday support/resistance, EMA20 filters intermediate trend,
# volume spike confirms breakout authenticity. Session filter (08-20 UTC) reduces noise trades.
# Discrete position size 0.20 to control risk and minimize fee churn.

name = "1h_Camarilla_R3_S3_Breakout_4hEMA20_VolumeSpike_v1"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar has no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 4h data for EMA20 trend filter (MTF)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA20 on 4h close
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align HTF arrays to 1h timeframe (wait for completed 4h bar)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate Camarilla levels from previous 4h bar
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low)
    #          S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_vals = df_4h['close'].values
    
    camarilla_r3_4h = close_4h_vals + 1.1 * (high_4h - low_4h)
    camarilla_s3_4h = close_4h_vals - 1.1 * (high_4h - low_4h)
    
    # Align HTF arrays to 1h timeframe (wait for completed 4h bar)
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    # Volume filter: current 1h volume > 1.5x 20-period average (spike confirmation)
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_1h)
    
    # Session filter: 08-20 UTC (reduce noise trades)
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(camarilla_r3_4h_aligned[i]) or 
            np.isnan(camarilla_s3_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_1h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Camarilla R3 AND price > 4h EMA20 AND volume spike AND session
            if close[i] > camarilla_r3_4h_aligned[i] and close[i] > ema20_4h_aligned[i] and volume_filter[i] and session_filter[i]:
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price < Camarilla S3 AND price < 4h EMA20 AND volume spike AND session
            elif close[i] < camarilla_s3_4h_aligned[i] and close[i] < ema20_4h_aligned[i] and volume_filter[i] and session_filter[i]:
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
            # EXIT LONG: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.0 * atr[i])
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
            # EXIT SHORT: trailing stop hit (2.0x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.0 * atr[i])
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