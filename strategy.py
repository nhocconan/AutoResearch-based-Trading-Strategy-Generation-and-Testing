#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h for balanced trade frequency and noise reduction.
- HTF: 12h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Volume: Current 6h volume > 1.5 * 20-period 6h volume MA to confirm institutional participation.
- Donchian Channel: 20-period high/low breakout for structural price levels.
- Entry: Long when price breaks above Donchian(20) high AND 12h EMA50 bullish AND volume spike.
         Short when price breaks below Donchian(20) low AND 12h EMA50 bearish AND volume spike.
- Exit: Opposite Donchian breakout (price < Donchian(20) low for long, price > Donchian(20) high for short)
        OR loss of volume confirmation.
- Signal size: 0.25 discrete to balance return and drawdown control.
- Target: 80-120 total trades over 4 years (20-30/year) for 6h timeframe.
This strategy captures medium-term momentum with trend filtering and volume confirmation,
avoiding false breakouts in choppy markets while participating in strong directional moves.
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
    
    # Calculate 6h Donchian Channel (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Calculate 6h volume MA (20-period)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    df_12h_close = df_12h['close'].values
    ema_12h = pd.Series(df_12h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 6h
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        dch_high = donchian_high[i]
        dch_low = donchian_low[i]
        vol_spike = volume_spike[i]
        ema_val = ema_12h_aligned[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if vol_spike:
                # Bullish: price breaks above Donchian high AND 12h EMA50 bullish (close > EMA)
                if curr_high > dch_high and curr_close > ema_val:
                    signals[i] = 0.25
                    position = 1
                # Bearish: price breaks below Donchian low AND 12h EMA50 bearish (close < EMA)
                elif curr_low < dch_low and curr_close < ema_val:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low OR loss of volume confirmation
            if curr_low < dch_low or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high OR loss of volume confirmation
            if curr_high > dch_high or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0