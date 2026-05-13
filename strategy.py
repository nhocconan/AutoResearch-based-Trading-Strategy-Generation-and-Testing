#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND 4h choppiness index > 61.8 (range).
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND 4h choppiness index > 61.8 (range).
# Uses ATR(14) trailing stop (2.0x) for risk control.
# Camarilla pivots provide precise intraday support/resistance levels that work in ranging markets.
# Choppiness index > 61.8 ensures we only trade in ranging conditions, avoiding false breakouts in strong trends.
# Volume spike confirms breakout legitimacy. Target: 80-160 total trades over 4 years (20-40/year) on 4h.

name = "4h_Camarilla_R3S3_Breakout_1dVolume_Chop_v1"
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
    # For 4h timeframe, we use 1d OHLC to calculate Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R3, R2, R1, PP, S1, S2, S3
    # PP = (high + low + close) / 3
    # Range = high - low
    # R3 = PP + Range * 1.1/2
    # S3 = PP - Range * 1.1/2
    pp = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    r3 = pp + (rang * 1.1 / 2.0)
    s3 = pp - (rang * 1.1 / 2.0)
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Calculate volume confirmation: volume > 2.0x 20-period average (1d)
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_confirm_1d = df_1d['volume'].values > (2.0 * vol_ma_20_1d)
    volume_confirm_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm_1d.astype(float))
    
    # Calculate 4h choppiness index (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * log10(highest_high - lowest_low))) / log10(n)
    # Simplified: CHOP = 100 * log10(sum(TR14) / (14 * log10(HH14 - LL14))) / log10(14)
    tr_4h = tr  # Already calculated TR for ATR
    sum_tr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_denominator = 14 * np.log10(highest_high_14 - lowest_low_14 + 1e-10)
    chop_denominator = np.where(chop_denominator <= 0, np.nan, chop_denominator)
    chop_ratio = sum_tr_14 / chop_denominator
    chop_ratio = np.where(chop_ratio <= 0, np.nan, chop_ratio)
    chop = 100 * np.log10(chop_ratio) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = np.full(n, np.nan)  # Track highest high since entry for longs
    lowest_since_entry = np.full(n, np.nan)   # Track lowest low since entry for shorts
    
    for i in range(100, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(volume_confirm_1d_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price > Camarilla R3 AND 1d volume confirmation AND chop > 61.8 (range)
            if close[i] > r3_aligned[i] and volume_confirm_1d_aligned[i] > 0.5 and chop[i] > 61.8:
                signals[i] = 0.25
                position = 1
                highest_since_entry[i] = high[i]  # Initialize tracking
            # SHORT: Price < Camarilla S3 AND 1d volume confirmation AND chop > 61.8 (range)
            elif close[i] < s3_aligned[i] and volume_confirm_1d_aligned[i] > 0.5 and chop[i] > 61.8:
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