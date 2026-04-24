#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1d for capturing multi-day trends.
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 1d volume > 2.0 * 20-period volume MA to capture institutional interest.
- Donchian: 20-period high/low breakout for entry signals.
- Entry: Long when price > Donchian(20) high AND 1w EMA50 bullish AND volume spike.
         Short when price < Donchian(20) low AND 1w EMA50 bearish AND volume spike.
- Exit: Opposite Donchian breakout (price < Donchian(20) low for long, price > Donchian(20) high for short).
- Signal size: 0.25 discrete to balance return and drawdown while minimizing fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
This strategy captures strong trending moves with institutional confirmation, avoiding counter-trend trades.
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
    
    # Calculate 1d Donchian(20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 20-period 1w volume MA
    df_1w_volume = df_1w['volume'].values
    vol_ma_1w = pd.Series(df_1w_volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Volume confirmation: current 1d volume > 2.0 * 20-period 1w volume MA (aligned)
    volume_spike = volume > (2.0 * vol_ma_1w_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Need enough bars for Donchian, EMA50, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish: Price breaks above Donchian high AND 1w EMA50 bullish (close > EMA)
                if curr_high > donchian_high[i] and curr_close > ema_1w_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish: Price breaks below Donchian low AND 1w EMA50 bearish (close < EMA)
                elif curr_low < donchian_low[i] and curr_close < ema_1w_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Price breaks below Donchian low
            if curr_low < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above Donchian high
            if curr_high > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0