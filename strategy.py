#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla pivot breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d targeting 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla pivot levels: calculated from prior 1d OHLC using standard formula.
- Entry: Long when price breaks above prior Camarilla H3 level AND 1w EMA50 bullish AND volume > 1.5 * volume MA(20).
         Short when price breaks below prior Camarilla L3 level AND 1w EMA50 bearish AND volume > 1.5 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below prior Camarilla H4 level,
        exit short when price crosses above prior Camarilla L4 level.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy captures medium-term breakouts aligned with the weekly trend, designed to work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 55:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Camarilla pivot levels from prior 1d OHLC
    # Standard Camarilla formula: H4 = close + 1.1*(high-low)*1.1/2, etc.
    # We calculate for prior day to avoid look-ahead
    prior_high = pd.Series(high).shift(1).values
    prior_low = pd.Series(low).shift(1).values
    prior_close = pd.Series(close).shift(1).values
    
    # Calculate range
    rng = prior_high - prior_low
    
    # Camarilla levels
    camarilla_h4 = prior_close + rng * 1.1 * 1.1 / 2
    camarilla_h3 = prior_close + rng * 1.1 / 2
    camarilla_l3 = prior_close - rng * 1.1 / 2
    camarilla_l4 = prior_close - rng * 1.1 * 1.1 / 2
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough bars for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(camarilla_h4[i]) or 
            np.isnan(camarilla_l4[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above prior Camarilla H3 level AND 1w EMA50 bullish AND volume confirmed
            if curr_close > camarilla_h3[i] and curr_close > ema_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior Camarilla L3 level AND 1w EMA50 bearish AND volume confirmed
            elif curr_close < camarilla_l3[i] and curr_close < ema_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below prior Camarilla H4 level
            if curr_close < camarilla_h4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above prior Camarilla L4 level
            if curr_close > camarilla_l4[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0