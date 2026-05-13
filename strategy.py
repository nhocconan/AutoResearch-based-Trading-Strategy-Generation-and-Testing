#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 (1d) AND close > 1d EMA34 AND volume > 2.0x average
# Short when price breaks below Camarilla S3 (1d) AND close < 1d EMA34 AND volume > 2.0x average
# Exit when price crosses Camarilla H3/L3 (mean reversion) OR trend reversal (price crosses 1d EMA34)
# Uses 4h timeframe (target: 75-200 total trades over 4 years = 19-50/year) with 1d trend filter for BTC/ETH resilience.
# Camarilla provides precise pivot levels; 1d EMA34 filters trend; volume spike confirms breakout authenticity.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
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
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3, H3, L3) on 1d data (using previous bar's OHLC to avoid look-ahead)
    if len(high_1d) >= 1:
        # Use previous bar's OHLC to calculate today's Camarilla levels
        prev_high = pd.Series(high_1d).shift(1).values
        prev_low = pd.Series(low_1d).shift(1).values
        prev_close = pd.Series(close_1d).shift(1).values
        
        # Calculate pivot point
        pivot = (prev_high + prev_low + prev_close) / 3
        range_hl = prev_high - prev_low
        
        # Camarilla levels
        r3 = pivot + range_hl * 1.1 / 4
        s3 = pivot - range_hl * 1.1 / 4
        h3 = pivot + range_hl * 1.1 / 6
        l3 = pivot - range_hl * 1.1 / 6
    else:
        r3 = np.full_like(high_1d, np.nan)
        s3 = np.full_like(low_1d, np.nan)
        h3 = np.full_like(high_1d, np.nan)
        l3 = np.full_like(low_1d, np.nan)
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Get 1d data for EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 4h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and volume
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Camarilla R3 AND close > 1d EMA34 AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Camarilla S3 AND close < 1d EMA34 AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Camarilla H3 (mean reversion) OR trend reversal (close < 1d EMA34)
            if close[i] < h3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Camarilla L3 (mean reversion) OR trend reversal (close > 1d EMA34)
            if close[i] > l3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals