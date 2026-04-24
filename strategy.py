#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation, and ATR trailing stop.
- Primary timeframe: 4h to target 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Donchian channels: Upper = 20-period high, Lower = 20-period low.
- Entry: Long when price breaks above Donchian Upper AND 1d EMA50 bullish AND volume > 1.5 * volume MA(20).
         Short when price breaks below Donchian Lower AND 1d EMA50 bearish AND volume > 1.5 * volume MA(20).
- Exit: ATR-based trailing stop - exit long when price < highest_high_since_entry - 2.5*ATR,
        exit short when price > lowest_low_since_entry + 2.5*ATR.
- Signal size: 0.30 discrete to balance return and drawdown while minimizing fee churn.
This strategy focuses on strong breakouts with institutional volume and trend alignment,
proven to work in both bull and bear markets via Donchian structure and volume confirmation.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    df_1d_close = df_1d['close'].values
    ema_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate ATR(14) for trailing stop
    tr1 = np.abs(high[1:] - low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align HTF indicators to 4h
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14, 20)  # Need enough bars for EMA50, Donchian, ATR, Vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
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
            
            # Long: Price breaks above Donchian Upper AND 1d EMA50 bullish AND volume confirmed
            if curr_close > donchian_upper[i] and curr_close > ema_1d_aligned[i] and vol_confirmed:
                signals[i] = 0.30
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            # Short: Price breaks below Donchian Lower AND 1d EMA50 bearish AND volume confirmed
            elif curr_close < donchian_lower[i] and curr_close < ema_1d_aligned[i] and vol_confirmed:
                signals[i] = -0.30
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
                signals[i] = 0.30
        elif position == -1:
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit when price > lowest_low + 2.5*ATR
            if curr_close > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirmation_v1"
timeframe = "4h"
leverage = 1.0