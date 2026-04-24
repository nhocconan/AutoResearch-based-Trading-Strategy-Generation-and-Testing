#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d to target 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian levels: Upper20 and Lower20 from prior 20-period high/low (using prior data to avoid look-ahead).
- Entry: Long when price breaks above prior Upper20 AND 1w EMA50 bullish AND volume > 1.5 * volume MA(50).
         Short when price breaks below prior Lower20 AND 1w EMA50 bearish AND volume > 1.5 * volume MA(50).
- Exit: Close-based reversal - exit long when price crosses below prior 10-period low,
        exit short when price crosses above prior 10-period high.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy targets medium-term breakouts aligned with weekly trend, designed to work in both bull and bear markets.
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
    
    # Calculate prior 1d Donchian Upper20 and Lower20 levels
    # Upper20 = max(high, lookback=20)
    # Lower20 = min(low, lookback=20)
    # Using prior 20 periods to avoid look-ahead
    lookback = 20
    upper_20 = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lower_20 = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    
    # Calculate prior 1d Donchian Upper10 and Lower10 for exit
    lookback_exit = 10
    upper_10 = pd.Series(high).rolling(window=lookback_exit, min_periods=lookback_exit).max().shift(1).values
    lower_10 = pd.Series(low).rolling(window=lookback_exit, min_periods=lookback_exit).min().shift(1).values
    
    # Calculate volume MA(50) for confirmation (using 1d data)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 55, 50)  # Need enough bars for EMA50, Donchian, and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(upper_10[i]) or np.isnan(lower_10[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above prior Upper20 AND 1w EMA50 bullish AND volume confirmed
            if curr_close > upper_20[i] and curr_close > ema_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior Lower20 AND 1w EMA50 bearish AND volume confirmed
            elif curr_close < lower_20[i] and curr_close < ema_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior Lower10 (trend weakening)
            if curr_close < lower_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior Upper10 (trend weakening)
            if curr_close > upper_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0