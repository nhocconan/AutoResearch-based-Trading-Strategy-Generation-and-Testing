#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 AND close > 1d EMA34 AND volume > 2.0x average
# Short when price breaks below Camarilla S3 AND close < 1d EMA34 AND volume > 2.0x average
# Exit when price crosses Camarilla H3/L3 (mean reversion) OR trend reversal (price crosses 1d EMA34)
# Uses 4h timeframe (target: 75-200 total trades over 4 years = 19-50/year) with daily trend filter for BTC/ETH resilience.
# Camarilla levels provide intraday support/resistance; EMA34 filters trend; volume spike confirms breakout authenticity.

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
    
    # Get 4h data for Camarilla calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels on 4h data (using previous bar's OHLC)
    if len(high_4h) >= 2:
        # Use previous bar's OHLC to avoid look-ahead
        prev_high = np.roll(high_4h, 1)
        prev_low = np.roll(low_4h, 1)
        prev_close = np.roll(close_4h, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        # Camarilla formulas: H3/L3, R3/S3
        rang = prev_high - prev_low
        h3 = prev_close + (rang * 1.1 / 4)
        l3 = prev_close - (rang * 1.1 / 4)
        r3 = prev_close + (rang * 1.1 / 2)
        s3 = prev_close - (rang * 1.1 / 2)
        h4 = prev_close + (rang * 1.1 / 2) * 1.166
        l4 = prev_close - (rang * 1.1 / 2) * 1.166
    else:
        h3 = np.full_like(high_4h, np.nan)
        l3 = np.full_like(low_4h, np.nan)
        r3 = np.full_like(high_4h, np.nan)
        s3 = np.full_like(low_4h, np.nan)
        h4 = np.full_like(high_4h, np.nan)
        l4 = np.full_like(low_4h, np.nan)
    
    # Align Camarilla levels to 4h timeframe (already aligned since calculated on 4h)
    h3_aligned = align_htf_to_ltf(prices, df_4h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_4h, l3)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    h4_aligned = align_htf_to_ltf(prices, df_4h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_4h, l4)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
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
            # LONG: price > R3 AND close > 1d EMA34 AND volume spike
            if close[i] > r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < S3 AND close < 1d EMA34 AND volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < H3 (mean reversion) OR trend reversal (close < 1d EMA34)
            if close[i] < h3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > L3 (mean reversion) OR trend reversal (close > 1d EMA34)
            if close[i] > l3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals