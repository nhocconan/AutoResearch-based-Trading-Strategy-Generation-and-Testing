#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above 1d Camarilla R3 AND close > 1d EMA34 AND volume > 2.0x average (12h)
# Short when price breaks below 1d Camarilla S3 AND close < 1d EMA34 AND volume > 2.0x average (12h)
# Exit when price crosses 1d Camarilla pivot (mean reversion) OR trend reversal (price crosses 1d EMA34)
# Uses 12h timeframe (target: 50-150 total trades over 4 years = 12-37/year) with 1d trend filter for BTC/ETH resilience.
# Camarilla provides clear structure; 1d EMA34 filters trend; volume spike confirms breakout authenticity.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary calculations (OHLC for Camarilla, volume for filter)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Camarilla levels (R3, S3, pivot) on 12h data using previous bar's OHLC
    if len(high_12h) >= 1:
        # Use previous bar's OHLC to avoid look-ahead
        prev_high = np.roll(high_12h, 1)
        prev_low = np.roll(low_12h, 1)
        prev_close = np.roll(close_12h, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_hl = prev_high - prev_low
        camarilla_r3 = pivot + (range_hl * 1.1 / 4)
        camarilla_s3 = pivot - (range_hl * 1.1 / 4)
        camarilla_pivot = pivot
    else:
        camarilla_r3 = np.full_like(high_12h, np.nan)
        camarilla_s3 = np.full_like(low_12h, np.nan)
        camarilla_pivot = np.full_like(high_12h, np.nan)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pivot)
    
    # Get 1d data for EMA34 trend filter (HTF as specified)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 12h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_12h > (2.0 * vol_ma_12h)
    volume_filter_aligned = align_htf_to_ltf(prices, df_12h, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pivot_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Camarilla R3 AND close > 1d EMA34 AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Camarilla S3 AND close < 1d EMA34 AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Camarilla pivot (mean reversion) OR trend reversal (close < 1d EMA34)
            if close[i] < camarilla_pivot_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Camarilla pivot (mean reversion) OR trend reversal (close > 1d EMA34)
            if close[i] > camarilla_pivot_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals