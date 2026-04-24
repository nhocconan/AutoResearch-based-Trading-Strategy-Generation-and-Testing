#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Power with 1w EMA200 trend filter and volume confirmation.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1w EMA200 for trend direction (bullish if close > EMA200, bearish if close < EMA200).
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close).
- Entry: Long when Bull Power > 0 AND 1w EMA200 bullish AND volume > 1.5 * volume MA(20).
         Short when Bear Power < 0 AND 1w EMA200 bearish AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when Bull Power <= 0,
        exit short when Bear Power >= 0.
- Signal size: 0.25 discrete to balance return and drawdown.
Elder Ray measures bull/bear power relative to EMA13, effective in both trending and ranging markets.
The 1w EMA200 filter ensures we only trade with the major trend, reducing whipsaws in bear markets.
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
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA200 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate EMA13 for Elder Ray (using 6h data)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray Power
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 6h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 60, 200)  # Need enough bars for EMA200 and EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Bull Power > 0 AND 1w EMA200 bullish AND volume confirmed
            if curr_bull_power > 0 and close[i] > ema_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND 1w EMA200 bearish AND volume confirmed
            elif curr_bear_power < 0 and close[i] < ema_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when Bull Power <= 0 (bull power fading)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when Bear Power >= 0 (bear power fading)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Power_1wEMA200_Trend_VolumeConfirmation_v1"
timeframe = "6h"
leverage = 1.0