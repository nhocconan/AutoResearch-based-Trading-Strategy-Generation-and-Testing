#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND price > 12h EMA50 AND volume > 1.6x 20-period average.
# Short when price breaks below Camarilla S3 AND price < 12h EMA50 AND volume > 1.6x 20-period average.
# Exit on ATR(14) trailing stop (2.0x). Uses 4h primary timeframe and 12h HTF for trend alignment.
# Designed for BTC/ETH with strict entry to avoid overtrading (target: 20-50 trades/year).
# Camarilla R3/S3 levels provide strong intraday support/resistance with lower false breakout rate than R4/S4.
# EMA50 on 12h filters intermediate trend, volume spike confirms breakout authenticity.

name = "4h_Camarilla_R3_S3_Breakout_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Get 12h data for EMA50 trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h close
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF arrays to 4h timeframe (wait for completed 12h bar)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from previous 12h bar (using 12h data for pivot calculation)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_vals = df_12h['close'].values
    
    camarilla_r3_12h = close_12h_vals + (1.1 * (high_12h - low_12h) / 2)  # R3 = C + 1.1*(H-L)/2
    camarilla_s3_12h = close_12h_vals - (1.1 * (high_12h - low_12h) / 2)  # S3 = C - 1.1*(H-L)/2
    
    # Align HTF arrays to 4h timeframe (wait for completed 12h bar)
    camarilla_r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # Volume filter: current 4h volume > 1.6x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.6 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_r3_12h_aligned[i]) or 
            np.isnan(camarilla_s3_12h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Camarilla R3 AND price > 12h EMA50 AND volume spike
            if close[i] > camarilla_r3_12h_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: price < Camarilla S3 AND price < 12h EMA50 AND volume spike
            elif close[i] < camarilla_s3_12h_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
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
                signals[i] = 0.25
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
                signals[i] = -0.25
                # Carry forward tracking
                if i > 0:
                    lowest_since_entry[i] = lowest_since_entry[i-1]
    
    return signals