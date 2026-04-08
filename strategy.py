#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_trend_v2
# Hypothesis: Camarilla pivot levels from daily timeframe act as institutional support/resistance.
# Long when price breaks above R4 with volume confirmation in uptrend (EMA50 > EMA200).
# Short when price breaks below S4 with volume confirmation in downtrend (EMA50 < EMA200).
# Uses 4h timeframe for entries and daily for pivot calculation to reduce noise.
# Volume filter requires current volume > 1.5x 20-period average to avoid false breakouts.
# Designed for fewer, higher-quality trades to minimize fee drag in ranging markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_trend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA50 and EMA200 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Volume average for confirmation
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's levels
    phigh = np.concatenate([[np.nan], high_1d[:-1]])
    plow = np.concatenate([[np.nan], low_1d[:-1]])
    pclose = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot = (phigh + plow + pclose) / 3
    range_ = phigh - plow
    
    # Camarilla levels
    r4 = pivot + (range_ * 1.1 / 2)
    r3 = pivot + (range_ * 1.1 / 4)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any data is NaN
        if (np.isnan(ema50[i]) or np.isnan(ema200[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(s4_aligned[i])):
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Long: price breaks above R4 with volume in uptrend
        if (close[i] > r4_aligned[i] and vol_confirm and 
            ema50[i] > ema200[i]):
            signals[i] = 0.25
        # Short: price breaks below S4 with volume in downtrend
        elif (close[i] < s4_aligned[i] and vol_confirm and 
              ema50[i] < ema200[i]):
            signals[i] = -0.25
        # Otherwise, stay flat (0)
        else:
            signals[i] = 0.0
    
    return signals