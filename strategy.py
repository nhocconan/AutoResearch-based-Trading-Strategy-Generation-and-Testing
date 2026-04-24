#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian channel: 20-period high/low from prior 4h candle (using prior close to avoid look-ahead).
- Entry: Long when price breaks above prior Donchian high AND 1w EMA50 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior Donchian low AND 1w EMA50 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 1w EMA50,
        exit short when price crosses above 1w EMA50.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures medium-term breakouts aligned with the 1w trend, designed to work in both bull and bear markets.
Using 1w trend instead of 1d may reduce whipsaw and improve generalization to bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate prior 4h Donchian(20) levels (using prior candle to avoid look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate volume MA(20) for confirmation (using 4h data)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 55)  # Need enough bars for EMA50 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior Donchian high AND 1w EMA50 bullish AND volume confirmed
            if curr_close > donchian_high[i] and curr_close > ema_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior Donchian low AND 1w EMA50 bearish AND volume confirmed
            elif curr_close < donchian_low[i] and curr_close < ema_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below 1w EMA50 (trend change)
            if curr_close < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above 1w EMA50 (trend change)
            if curr_close > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0