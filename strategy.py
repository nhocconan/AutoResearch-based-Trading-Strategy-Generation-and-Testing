#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d to target 30-100 total trades over 4 years (7-25/year).
- HTF: 1w EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla levels: H3 = prior_1w_close + 1.1*(prior_1w_high - prior_1w_low)*1.1/12, 
                     L3 = prior_1w_close - 1.1*(prior_1w_high - prior_1w_low)*1.1/12.
- Entry: Long when price breaks above Camarilla H3 AND 1w EMA50 bullish AND volume > 1.5 * volume MA(20).
         Short when price breaks below Camarilla L3 AND 1w EMA50 bearish AND volume > 1.5 * volume MA(20).
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 2.5*ATR,
        exit short when price > lowest_low_since_entry + 2.5*ATR.
- Signal size: 0.25 discrete to balance return and drawdown.
This strategy targets fewer, higher-quality breakouts on the 1d timeframe with institutional volume confirmation
and weekly trend alignment, reducing fee drag while maintaining profitability in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    df_1w_close = df_1w['close'].values
    ema_1w = pd.Series(df_1w_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate prior week's Camarilla H3 and L3 levels
    # H3 = prior_week_close + 1.1*(prior_week_high - prior_week_low)*1.1/12
    # L3 = prior_week_close - 1.1*(prior_week_high - prior_week_low)*1.1/12
    prior_close = df_1w['close'].shift(1).values
    prior_high = df_1w['high'].shift(1).values
    prior_low = df_1w['low'].shift(1).values
    camarilla_h3 = prior_close + 1.1 * (prior_high - prior_low) * 1.1 / 12
    camarilla_l3 = prior_close - 1.1 * (prior_high - prior_low) * 1.1 / 12
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 1d
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50, 14, 20)  # Need enough bars for EMA50, ATR, Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: Price breaks above Camarilla H3 AND 1w EMA50 bullish AND volume confirmed
            if curr_close > camarilla_h3_aligned[i] and curr_close > ema_1w_aligned[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Price breaks below Camarilla L3 AND 1w EMA50 bearish AND volume confirmed
            elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_1w_aligned[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.5*ATR
            if curr_close < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.5*ATR
            if curr_close > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0