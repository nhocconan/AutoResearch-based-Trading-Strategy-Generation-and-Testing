#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (EMA34) and volume confirmation (1.5x MA20) to capture institutional reversal levels.
# Uses Camarilla pivot levels (R3/S3) from daily data as strong support/resistance where price often reverses or accelerates.
# Enters long when price breaks above R3 with 1d bullish trend (close > EMA34) and volume > 1.5x MA20.
# Enters short when price breaks below S3 with 1d bearish trend (close < EMA34) and volume > 1.5x MA20.
# Exits when price reverts to the Camarilla midpoint (R3/S3 midpoint) or opposite level (S3/R3) for mean reversion.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Camarilla levels provide high-probability institutional reaction points, effective in both trending and ranging markets.
# Volume filter reduces false breakouts, improving signal quality in low-liquidity conditions.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v1"
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from 1d data: R3, S3, and midpoint
    # Camarilla formula: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Actually: R3 = close + (high-low)*1.1/4, S3 = close - (high-low)*1.1/4
    # Midpoint = (R3 + S3)/2 = close
    diff_1d = high_1d - low_1d
    camarilla_r3 = close_1d + (diff_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (diff_1d * 1.1 / 4)
    camarilla_mid = close_1d  # (R3 + S3)/2 simplifies to close
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # Track entry price for exit logic
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla midpoint (mean reversion) or breaks below S3 (failure)
            if close[i] < camarilla_mid_aligned[i] or close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla midpoint (mean reversion) or breaks above R3 (failure)
            if close[i] > camarilla_mid_aligned[i] or close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]
    
    return signals