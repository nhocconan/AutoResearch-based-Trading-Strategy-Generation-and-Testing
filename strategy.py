#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume spike (>2x 20-period average), and choppiness regime filter (CHOP > 61.8 for ranging markets). 
# Long when price breaks above Camarilla R3, 1d EMA34 is rising, volume > 2x average, and CHOP > 61.8 (range). 
# Short when price breaks below Camarilla S3, 1d EMA34 is falling, volume > 2x average, and CHOP > 61.8. 
# Uses ATR(14) trailing stop (2.5x) for risk control. 
# Discrete position sizing (0.25) to minimize fee churn. 
# Target: 75-200 total trades over 4 years (19-50/year) on 4h. 
# Designed to work in both bull (trend filter) and bear (range reversion via CHOP filter) markets.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_Chop_v1"
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
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1d data for EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14) over 14 periods) / (log10(highest high - lowest low over 14 periods)))
    # We'll use a simplified version: CHOP = 100 * log10(sum(tr) over 14 periods) / log10(highest high - lowest low over 14 periods)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_low_diff = highest_high_14 - lowest_low_14
    # Avoid division by zero and log of zero
    chop = np.zeros(n)
    for i in range(n):
        if highest_low_diff[i] > 0 and sum_tr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_tr_14[i]) / np.log10(highest_low_diff[i])
        else:
            chop[i] = 50  # Neutral value when undefined
    chop_filter = chop > 61.8  # Indicates ranging market (good for mean reversion)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Camarilla R3 AND 1d EMA34 rising (trending up) AND volume confirmation AND chop filter (range)
            if close[i] > camarilla_r3_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_confirm[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Camarilla S3 AND 1d EMA34 falling (trending down) AND volume confirmation AND chop filter (range)
            elif close[i] < camarilla_s3_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_confirm[i] and chop_filter[i]:
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
            # EXIT LONG: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] < (highest_since_entry[i] - 2.5 * atr[i])
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
            # EXIT SHORT: trailing stop hit (2.5x ATR)
            trailing_stop = close[i] > (lowest_since_entry[i] + 2.5 * atr[i])
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