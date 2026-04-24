#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 12h to target 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA34 for trend direction (bullish if close > EMA34, bearish if close < EMA34).
- Camarilla levels: H3 and L3 from prior 12h bar (using prior close to avoid look-ahead).
- Entry: Long when price breaks above prior H3 AND 1d EMA34 bullish AND volume > 2.0 * volume MA(20).
         Short when price breaks below prior L3 AND 1d EMA34 bearish AND volume > 2.0 * volume MA(20).
- Exit: Close-based reversal - exit long when price crosses below 1d EMA34,
        exit short when price crosses above 1d EMA34.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy targets mean reversion failures in strong trends, designed to work in both bull and bear markets
by aligning with the 1d trend and using volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate prior 12h Camarilla H3 and L3 levels
    # Typical price for prior bar
    typical_price = (high + low + close) / 3
    # Shift by 1 to use prior bar's typical price (avoid look-ahead)
    prior_typical = pd.Series(typical_price).shift(1).values
    # Prior bar's high and low
    prior_high = pd.Series(high).shift(1).values
    prior_low = pd.Series(low).shift(1).values
    # Camarilla H3 and L3
    camarilla_h3 = prior_typical + (prior_high - prior_low) * 1.1 / 4
    camarilla_l3 = prior_typical - (prior_high - prior_low) * 1.1 / 4
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 12h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 20)  # Need enough bars for EMA34 and calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(camarilla_h3[i]) or 
            np.isnan(camarilla_l3[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (2.0x threshold)
            vol_confirmed = curr_volume > 2.0 * vol_ma[i]
            
            # Long: Price breaks above prior H3 AND 1d EMA34 bullish AND volume confirmed
            if curr_close > camarilla_h3[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below prior L3 AND 1d EMA34 bearish AND volume confirmed
            elif curr_close < camarilla_l3[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price crosses below 1d EMA34 (trend change)
            if curr_close < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price crosses above 1d EMA34 (trend change)
            if curr_close > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H3L3_1dEMA34_Trend_VolumeConfirmation_v1"
timeframe = "12h"
leverage = 1.0