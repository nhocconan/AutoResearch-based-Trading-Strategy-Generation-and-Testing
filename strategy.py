#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d volume spike and 1d EMA34 trend filter.
# Long when price breaks above R3 with volume spike and close > EMA34 (1d).
# Short when price breaks below S3 with volume spike and close < EMA34 (1d).
# Exit when price reverts to the 1d EMA34 or ATR(14) < ATR(50) * 0.8 (low vol exit).
# Uses discrete position sizing (0.25) to limit fee churn.
# Designed for low trade frequency (~12-37/year) by requiring confluence of breakout, volume, and trend.
# Camarilla levels provide intraday support/resistance; breakouts with volume indicate institutional participation.
# Effective in both bull and bear markets by capturing strong directional moves with trend and volatility filters.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "6h"
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) and ATR(50) on 1d for volatility regime
    tr1 = np.maximum(high_1d - low_1d, np.absolute(high_1d - np.roll(close_1d, 1)))
    tr2 = np.absolute(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high_1d[0] - low_1d[0]  # first bar
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50_1d = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    # Volume spike: current volume > 2.0 * 20-period average volume (1d)
    vol_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    volume_spike = volume_1d > (2.0 * vol_ma20)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    # Calculate Camarilla levels for 1d: based on previous day's OHLC
    # R4 = close + 1.5 * (high - low)
    # R3 = close + 1.1 * (high - low)
    # S3 = close - 1.1 * (high - low)
    # S4 = close - 1.5 * (high - low)
    camarilla_r3 = close_1d + 1.1 * (high_1d - low_1d)
    camarilla_s3 = close_1d - 1.1 * (high_1d - low_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Track entry price for stoploss (optional, using signal reversal as primary exit)
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i]) or \
           np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(volume_spike_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3, volume spike, and close > EMA34 (1d)
            if close[i] > camarilla_r3_aligned[i] and volume_spike_aligned[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: price breaks below S3, volume spike, and close < EMA34 (1d)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike_aligned[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price reverts to EMA34 OR low volatility regime (ATR14 < ATR50 * 0.8)
            if close[i] <= ema34_1d_aligned[i] or atr14_1d_aligned[i] < atr50_1d_aligned[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: price reverts to EMA34 OR low volatility regime (ATR14 < ATR50 * 0.8)
            if close[i] >= ema34_1d_aligned[i] or atr14_1d_aligned[i] < atr50_1d_aligned[i] * 0.8:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals