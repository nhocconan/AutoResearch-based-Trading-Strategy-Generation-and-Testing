#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level, 4h EMA50 is rising (trending up), and volume > 1.8x 20-period average.
# Short when price breaks below Camarilla S3 level, 4h EMA50 is falling (downtrend), and volume > 1.8x 20-period average.
# Uses ATR(14) trailing stop (1.5x) for risk control.
# Session filter: only trade 08-20 UTC to avoid low-volume periods.
# Target: 60-150 total trades over 4 years (15-37/year) on 1h by using 4h for signal direction and 1h only for entry timing.
# Discrete position size: 0.20 to minimize fee churn and manage drawdown.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_Volume_v1"
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
    
    # Calculate Camarilla levels (based on previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, S3
    # R3 = close + 1.1*(high-low)/4
    # S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d) / 4
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA(50) on 4h data
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h timeframe (wait for 4h bar to close)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Camarilla R3 AND 4h EMA50 rising (trending up) AND volume confirmation AND in session
            if close[i] > camarilla_r3_aligned[i] and ema_50_4h_aligned[i] > ema_50_4h_aligned[i-1] and volume_confirm[i] and in_session[i]:
                signals[i] = 0.20
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Camarilla S3 AND 4h EMA50 falling (trending down) AND volume confirmation AND in session
            elif close[i] < camarilla_s3_aligned[i] and ema_50_4h_aligned[i] < ema_50_4h_aligned[i-1] and volume_confirm[i] and in_session[i]:
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
            # EXIT LONG: trailing stop hit (1.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 1.5 * atr[i])
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
            # EXIT SHORT: trailing stop hit (1.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 1.5 * atr[i])
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