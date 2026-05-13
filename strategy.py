#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND 1w EMA50 is rising AND volume > 2.0x 24-period average.
# Short when price breaks below Camarilla S3 AND 1w EMA50 is falling AND volume > 2.0x 24-period average.
# Uses ATR(14) trailing stop (2.0x) for risk control.
# Uses discrete position sizing (0.25) to minimize fee drag and manage drawdown.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h.

name = "12h_Camarilla_R3_S3_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
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
    
    # Calculate Camarilla pivot levels (R3, S3) on 12h data
    # Pivot = (high + low + close) / 3
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    pivot = (high + low + close) / 3.0
    camarilla_r3 = close + (high - low) * 1.1 / 4.0
    camarilla_s3 = close - (high - low) * 1.1 / 4.0
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe (wait for 1w bar to close)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate volume confirmation: volume > 2.0x 24-period average (24*12h = 12 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Camarilla R3 AND 1w EMA50 rising (trending up) AND volume spike
            if close[i] > camarilla_r3[i] and ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Camarilla S3 AND 1w EMA50 falling (trending down) AND volume spike
            elif close[i] < camarilla_s3[i] and ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and volume_confirm[i]:
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