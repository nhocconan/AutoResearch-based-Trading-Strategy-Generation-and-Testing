#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation.
- Primary timeframe: 1h to target 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Camarilla levels: H3 = prior_4h_close + 1.1*(prior_4h_high - prior_4h_low)*1.1/12,
                     L3 = prior_4h_close - 1.1*(prior_4h_high - prior_4h_low)*1.1/12.
- Entry: Long when price breaks above Camarilla H3 AND 4h EMA50 bullish AND volume > 1.3 * volume MA(20).
         Short when price breaks below Camarilla L3 AND 4h EMA50 bearish AND volume > 1.3 * volume MA(20).
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 2.0*ATR,
        exit short when price > lowest_low_since_entry + 2.0*ATR.
- Signal size: 0.20 discrete to minimize fee churn.
- Session filter: Only trade between 08:00-20:00 UTC to avoid low-volume noisy periods.
This strategy uses 4h for signal direction and 1h for precise entry timing, reducing whipsaw while capturing institutional breakouts.
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
    
    # Get 4h data for EMA50 trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    df_4h_close = df_4h['close'].values
    ema_4h = pd.Series(df_4h_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate prior 4h bar's Camarilla H3 and L3 levels
    # H3 = prior_4h_close + 1.1*(prior_4h_high - prior_4h_low)*1.1/12
    # L3 = prior_4h_close - 1.1*(prior_4h_high - prior_4h_low)*1.1/12
    prior_close = df_4h['close'].shift(1).values
    prior_high = df_4h['high'].shift(1).values
    prior_low = df_4h['low'].shift(1).values
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
    
    # Align HTF indicators to 1h
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3)
    
    # Pre-compute session filter (08:00-20:00 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 50, 14, 20)  # Need enough bars for EMA50, ATR, Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or
            not in_session[i]):
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
            vol_confirmed = curr_volume > 1.3 * vol_ma[i]
            
            # Long: Price breaks above Camarilla H3 AND 4h EMA50 bullish AND volume confirmed
            if curr_close > camarilla_h3_aligned[i] and curr_close > ema_4h_aligned[i] and vol_confirmed:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Price breaks below Camarilla L3 AND 4h EMA50 bearish AND volume confirmed
            elif curr_close < camarilla_l3_aligned[i] and curr_close < ema_4h_aligned[i] and vol_confirmed:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit when price < highest_high - 2.0*ATR
            if curr_close < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.0*ATR
            if curr_close > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_4hEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1h"
leverage = 1.0