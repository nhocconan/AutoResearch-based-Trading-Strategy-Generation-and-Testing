#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation.
# Uses 20-day Donchian channels for breakout signals. Long when price breaks above upper band
# with weekly EMA up and volume confirmation. Short when price breaks below lower band
# with weekly EMA down and volume confirmation. Designed to capture trends while filtering
# false breakouts with weekly trend and volume. Targets 15-25 trades/year to minimize fee drag.
name = "1d_Donchian20_1wEMA_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    donchian_window = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_window - 1, n):
        upper[i] = np.max(high[i - donchian_window + 1:i + 1])
        lower[i] = np.min(low[i - donchian_window + 1:i + 1])
    
    # 1w EMA trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band + weekly EMA up + volume
            if price > upper[i] and ema_1w_aligned[i] > ema_1w_aligned[i-1] and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + weekly EMA down + volume
            elif price < lower[i] and ema_1w_aligned[i] < ema_1w_aligned[i-1] and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below lower Donchian band or weekly EMA turns down
            if price < lower[i] or ema_1w_aligned[i] < ema_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above upper Donchian band or weekly EMA turns up
            if price > upper[i] or ema_1w_aligned[i] > ema_1w_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals