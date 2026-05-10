#!/usr/bin/env python3
# 4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Tight_v4
# Hypothesis: Further tighten entry conditions by requiring volume > 3x MA and price close outside Bollinger Bands (20,2) to avoid whipsaws.
# Uses Bollinger Band squeeze as volatility filter and requires confluence of price > upper band for longs, < lower band for shorts.
# Maintains Camarilla R1/S1 breakout with 1d EMA34 trend filter. Targets 10-20 trades/year to minimize fee drag.
# Designed to work in both bull and bear markets by aligning with higher timeframe trend and avoiding low-probability setups.

name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume_Tight_v4"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate ATR for volatility filter (using 14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Bollinger Bands (20,2) for volatility squeeze and entry filter
    close_series = pd.Series(close)
    bb_mid = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    
    # Bollinger Band squeeze: narrow bands indicate low volatility, wait for expansion
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    volatility_expansion = bb_width > bb_width_ma  # True when volatility is expanding
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate R1 and S1 (tighter levels)
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 6
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1d data for trend filter (EMA34)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume average for confirmation (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Warmup for Bollinger Bands, BB width MA, 1d EMA, and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(bb_width_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Strong volume confirmation and volatility expansion filter
        volume_confirm = volume[i] > volume_ma[i] * 3.0  # Increased threshold to reduce trades
        volatility_filter = volatility_expansion[i] and atr[i] > 0
        
        if position == 0:
            # Long entry: price breaks above R1 AND above BB upper band with volume confirmation, 1d uptrend, and volatility expansion
            if close[i] > r1_aligned[i] and close[i] > bb_upper[i] and volume_confirm and uptrend and volatility_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below S1 AND below BB lower band with volume confirmation, 1d downtrend, and volatility expansion
            elif close[i] < s1_aligned[i] and close[i] < bb_lower[i] and volume_confirm and downtrend and volatility_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below R1 or trend turns down or price closes below BB mid
            if close[i] < r1_aligned[i] or not uptrend or close[i] < bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above S1 or trend turns up or price closes above BB mid
            if close[i] > s1_aligned[i] or not downtrend or close[i] > bb_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals