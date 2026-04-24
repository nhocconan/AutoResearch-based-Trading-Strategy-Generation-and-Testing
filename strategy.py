#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
- Primary timeframe: 12h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian channel: 20-period high/low from prior 12h candle to avoid look-ahead.
- Entry: Long when price breaks above prior 20-bar 12h high AND 1d EMA50 bullish AND volume > 1.5 * volume MA(50).
         Short when price breaks below prior 20-bar 12h low AND 1d EMA50 bearish AND volume > 1.5 * volume MA(50).
- Exit: Close-based reversal - exit long when price crosses below 1d EMA50,
        exit short when price crosses above 1d EMA50.
- Signal size: 0.25 discrete to minimize fee churn and control drawdown.
This strategy captures medium-term breakouts aligned with the 1d trend, designed to work in both bull and bear markets.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate volume MA(50) for confirmation (using 12h data)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 50)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Calculate Donchian(20) from prior 20 bars (excluding current)
        if i >= 20:
            lookback_start = i - 20
            lookback_end = i  # exclude current bar
            period_high = np.max(high[lookback_start:lookback_end])
            period_low = np.min(low[lookback_start:lookback_end])
        else:
            # Not enough lookback data
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirmed = curr_volume > 1.5 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            # Long: Price breaks above prior 20-bar high AND 1d EMA50 bullish AND volume confirmed
            if curr_close > period_high and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior 20-bar low AND 1d EMA50 bearish AND volume confirmed
            elif curr_close < period_low and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below 1d EMA50 (trend change)
            if curr_close < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above 1d EMA50 (trend change)
            if curr_close > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0