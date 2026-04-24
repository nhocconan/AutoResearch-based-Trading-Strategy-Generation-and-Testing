#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA21 trend filter and volume spike filter.
- Primary timeframe: 1h targeting 60-150 total trades over 4 years (15-37/year).
- HTF: 4h EMA21 for trend filter (price > EMA21 = uptrend, price < EMA21 = downtrend).
- Camarilla levels from 1h: H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low).
- Entry: Long when close breaks above H3 AND price > 4h EMA21 AND volume > 2.5 * 1h volume MA(30);
         Short when close breaks below L3 AND price < 4h EMA21 AND volume > 2.5 * 1h volume MA(30).
- Exit: ATR-based trailing stop (2.0 * ATR(14)) from highest high/lowest low since entry.
- Signal size: 0.20 discrete to control fee drag and allow for multiple positions.
- Uses Camarilla structure for institutional levels, 4h EMA21 for trend alignment,
  volume spike for conviction. Designed to work in both bull (longs) and bear (shorts).
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
    
    # Get 4h data for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Calculate 1h Camarilla levels: H3 and L3
    camarilla_h3 = close + 1.125 * (high - low)
    camarilla_l3 = close - 1.125 * (high - low)
    
    # Calculate ATR(14) for 1h timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(30) for 1h timeframe
    vol_ma_1h = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(21, 30, 14)  # EMA21 needs 21, volume MA needs 30, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_21_aligned[i]) or 
            np.isnan(atr14[i]) or 
            np.isnan(vol_ma_1h[i])):
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
        
        # Volume confirmation: 2.5x threshold for controlled entry frequency
        vol_confirm = curr_volume > 2.5 * vol_ma_1h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Close breaks above H3 AND price > 4h EMA21 (uptrend)
                if curr_close > camarilla_h3[i] and curr_close > ema_21_aligned[i]:
                    signals[i] = 0.20
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Close breaks below L3 AND price < 4h EMA21 (downtrend)
                elif curr_close < camarilla_l3[i] and curr_close < ema_21_aligned[i]:
                    signals[i] = -0.20
                    position = -1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
        elif position == 1:
            # Long position: update highest since entry and check exit
            highest_since_entry = max(highest_since_entry, curr_high)
            lowest_since_entry = min(lowest_since_entry, curr_low)
            
            # Stoploss: 2.0 * ATR below highest high since entry
            stoploss = highest_since_entry - 2.0 * curr_atr
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
            
            # Stoploss: 2.0 * ATR above lowest low since entry
            stoploss = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stoploss:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA21_Trend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0