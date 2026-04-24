#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3L3 breakout with 4h EMA50 trend filter and volume spike confirmation.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when price breaks above H3 level AND price > 4h EMA50 AND volume > 2.0 * 1h volume MA(20);
         Short when price breaks below L3 level AND price < 4h EMA50 AND volume > 2.0 * 1h volume MA(20).
- Exit: ATR-based trailing stop (2.5 * ATR(14)) from highest high/lowest low since entry.
- Signal size: 0.20 discrete to control fee drag.
- Uses 4h/1d for signal direction, 1h only for entry timing.
- Session filter: 08-20 UTC to reduce noise trades.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours for filtering
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4f['low'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla pivot levels for 4h timeframe using prior 4h OHLC
    # Need prior 4h bar's OHLC, so shift by 1
    prev_close_4h = np.concatenate([[close_4h[0]], close_4h[:-1]])
    prev_high_4h = np.concatenate([[high_4h[0]], high_4h[:-1]])
    prev_low_4h = np.concatenate([[low_4h[0]], low_4h[:-1]])
    
    camarilla_H3 = prev_close_4h + 1.25 * (prev_high_4h - prev_low_4h)
    camarilla_L3 = prev_close_4h - 1.25 * (prev_high_4h - prev_low_4h)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_L3)
    
    # Calculate volume MA(20) for 1h timeframe
    vol_ma_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for 1h timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(vol_ma_1h[i]) or 
            np.isnan(atr14[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr14[i]
        
        # Volume confirmation: 2.0x threshold for strict entry
        vol_confirm = curr_volume > 2.0 * vol_ma_1h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above H3 AND price > 4h EMA50 (uptrend)
                if curr_close > camarilla_H3_aligned[i] and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Price breaks below L3 AND price < 4h EMA50 (downtrend)
                elif curr_close < camarilla_L3_aligned[i] and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
        elif position == 1:
            # Long position: update highest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.5 * ATR below highest high since entry
            stoploss = highest_since_entry - 2.5 * curr_atr
            if curr_close < stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: update lowest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.5 * ATR above lowest low since entry
            stoploss = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0