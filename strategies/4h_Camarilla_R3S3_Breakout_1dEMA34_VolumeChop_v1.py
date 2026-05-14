#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter, volume confirmation (>2.0x 20-period average), and choppiness regime filter (CHOP > 61.8 for mean reversion, < 38.2 for trend following). Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Designed to work in both bull and bear markets by combining price structure (Camarilla), trend (1d EMA34), volume strength, and regime awareness.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeChop_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Volume confirmation: > 2.0x 20-period average (higher threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    # Choppiness Index (CHOP) - regime filter
    atr_period = 14
    chop_period = 14
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    max_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    chop = 100 * np.log10((atr * np.sqrt(chop_period)) / (max_high - min_low)) / np.log10(chop_period)
    chop = np.where((max_high - min_low) == 0, 50, chop)  # avoid division by zero
    
    chop_range = chop > 61.8   # ranging market (mean reversion)
    chop_trend = chop < 38.2   # trending market (trend following)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34) - trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_confirm[i]) or
            np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla levels for today (using previous bar's OHLC)
        if i >= 1:
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            range_ = prev_high - prev_low
            
            # Camarilla levels (R3/S3 = standard levels)
            R3 = prev_close + range_ * 1.1 / 4
            S3 = prev_close - range_ * 1.1 / 4
        else:
            R3 = np.nan
            S3 = np.nan
        
        if position == 0:
            # LONG: Price breaks above R3 AND close > 1d EMA34 (bullish trend) AND volume confirm AND (choppy: mean reversion OR trending: momentum)
            if (not np.isnan(R3) and 
                close[i] > R3 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_confirm[i] and
                (chop_range[i] or chop_trend[i])):  # allow both regimes but with different logic implicit in entry
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 AND close < 1d EMA34 (bearish trend) AND volume confirm AND (choppy: mean reversion OR trending: momentum)
            elif (not np.isnan(S3) and 
                  close[i] < S3 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_confirm[i] and
                  (chop_range[i] or chop_trend[i])):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below 1d EMA34 (trend change) OR touches S3 (mean reversion in chop)
            if close[i] < ema_34_1d_aligned[i] or (chop_range[i] and close[i] < S3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above 1d EMA34 (trend change) OR touches R3 (mean reversion in chop)
            if close[i] > ema_34_1d_aligned[i] or (chop_range[i] and close[i] > R3):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals