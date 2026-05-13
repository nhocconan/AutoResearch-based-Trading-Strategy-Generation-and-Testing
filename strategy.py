#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 AND close > 1d EMA34 AND volume > 2.0x average
# Short when price breaks below S3 AND close < 1d EMA34 AND volume > 2.0x average
# Exit when price crosses Camarilla H3/L3 (mean reversion) OR trend reversal (close crosses 1d EMA34)
# Uses 4h timeframe (target: 75-200 total trades over 4 years = 19-50/year) with daily trend filter for BTC/ETH resilience.
# Daily EMA34 provides strong trend filter reducing whipsaw; volume spike confirms breakout authenticity.
# Camarilla levels calculated from prior 1d OHLC, providing institutional support/resistance levels.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume_v1"
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
    
    # Get 4h data for primary timeframe calculations
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Get 1d data for Camarilla levels (OHLC from prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate Camarilla levels for today using prior day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But standard Camarilla uses: R3 = close + 1.1*(high-low), S3 = close - 1.1*(high-low)
    # H3/L3 are closer to the mean: H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Actually, standard Camarilla levels:
    # R4 = close + 1.5*(high-low)
    # R3 = close + 1.1*(high-low)
    # R2 = close + 0.55*(high-low)
    # R1 = close + 0.275*(high-low)
    # PP = (high+low+close)/3
    # S1 = close - 0.275*(high-low)
    # S2 = close - 0.55*(high-low)
    # S3 = close - 1.1*(high-low)
    # S4 = close - 1.5*(high-low)
    
    # Calculate prior day's range
    prev_high = np.roll(high_1d, 1)  # yesterday's high
    prev_low = np.roll(low_1d, 1)    # yesterday's low
    prev_close = np.roll(close_1d, 1) # yesterday's close
    
    # First day has no prior data
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels based on prior day
    rang = prev_high - prev_low
    R3 = prev_close + 1.1 * rang
    S3 = prev_close - 1.1 * rang
    H3 = prev_close + 1.1 * rang / 2  # H3 is midpoint between R3 and close
    L3 = prev_close - 1.1 * rang / 2  # L3 is midpoint between S3 and close
    
    # Get 1d data for EMA34 trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all 1d data to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current 4h volume > 2.0x 20-period average (spike confirmation)
    vol_ma_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume_4h > (2.0 * vol_ma_4h)
    
    # Align volume filter to 4h (already on 4h)
    # No need to align since volume_4h is already 4h data
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for EMA and volume MA
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > R3 AND close > 1d EMA34 AND volume spike
            if close[i] > R3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < S3 AND close < 1d EMA34 AND volume spike
            elif close[i] < S3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < H3 (mean reversion) OR trend reversal (close < 1d EMA34)
            if close[i] < H3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > L3 (mean reversion) OR trend reversal (close > 1d EMA34)
            if close[i] > L3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals