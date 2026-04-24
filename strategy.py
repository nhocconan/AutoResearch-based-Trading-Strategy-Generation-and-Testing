#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d for lower trade frequency and reduced fee drag.
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 1d volume > 2.0 * 20-period 1d volume MA to capture institutional interest.
- Entry: Long when price breaks above Donchian(20) upper band AND 1w EMA50 bullish AND volume spike.
         Short when price breaks below Donchian(20) lower band AND 1w EMA50 bearish AND volume spike.
- Exit: Opposite Donchian breakout (price crosses middle band) or loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown.
- Target: 50-100 total trades over 4 years (12-25/year) for 1d timeframe.
This strategy captures medium-term trends with institutional volume confirmation, avoiding whipsaws in choppy markets.
Works in both bull and bear markets by only taking trades in the direction of the 1w trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channel (20-period)
    lookback = 20
    upper_band = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_band = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    middle_band = (upper_band + lower_band) / 2.0
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1d volume MA
    vol_ma_1d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: current 1d volume > 2.0 * 20-period 1d volume MA
    volume_spike = volume > (2.0 * vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, 50, 20)  # Need enough bars for Donchian, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_val = ema_1w_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Price breaks above upper band AND 1w EMA50 bullish (close > EMA)
                if curr_high > upper_band[i] and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Price breaks below lower band AND 1w EMA50 bearish (close < EMA)
                elif curr_low < lower_band[i] and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Price crosses below middle band OR loss of volume confirmation
            if curr_close < middle_band[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above middle band OR loss of volume confirmation
            if curr_close > middle_band[i] or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0