#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3L3 breakout with 1d EMA34 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA34 for trend filter (price > EMA34 = uptrend, price < EMA34 = downtrend).
- Entry: Long when price breaks above H3 level AND price > 1d EMA34 AND volume > 2.0 * 4h volume MA(20);
         Short when price breaks below L3 level AND price < 1d EMA34 AND volume > 2.0 * 4h volume MA(20).
- Exit: ATR-based trailing stop (2.5 * ATR(14)) from highest high/lowest low since entry.
- Signal size: 0.25 discrete to control fee drag.
- Camarilla pivot levels provide mathematical support/resistance based on prior day's range.
  H3 and L3 are strong intraday levels where breakouts often indicate continuation.
  Combined with 1d EMA34 trend filter and volume confirmation, this strategy captures
  strong intraday momentum in both bull and bear markets while avoiding false breakouts.
  EMA34 is more responsive than EMA50, potentially capturing trends earlier with fewer whipsaws.
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels for 4h timeframe using prior 1d OHLC
    # H4 = close + 1.5 * (high - low)
    # H3 = close + 1.25 * (high - low)
    # H2 = close + 1.166 * (high - low)
    # H1 = close + 1.0833 * (high - low)
    # L1 = close - 1.0833 * (high - low)
    # L2 = close - 1.166 * (high - low)
    # L3 = close - 1.25 * (high - low)
    # L4 = close - 1.5 * (high - low)
    
    # We need prior day's OHLC, so shift by 1
    prev_close_1d = np.concatenate([[close_1d[0]], close_1d[:-1]])
    prev_high_1d = np.concatenate([[high_1d[0]], high_1d[:-1]])
    prev_low_1d = np.concatenate([[low_1d[0]], low_1d[:-1]])
    
    camarilla_H3 = prev_close_1d + 1.25 * (prev_high_1d - prev_low_1d)
    camarilla_L3 = prev_close_1d - 1.25 * (prev_high_1d - prev_low_1d)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for 4h timeframe
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
    start_idx = max(50, 20, 14)  # EMA34 needs 50 (we use 34 but keep buffer), volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(camarilla_H3_aligned[i]) or 
            np.isnan(camarilla_L3_aligned[i]) or 
            np.isnan(vol_ma_4h[i]) or 
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
        vol_confirm = curr_volume > 2.0 * vol_ma_4h[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Price breaks above H3 AND price > 1d EMA34 (uptrend)
                if curr_close > camarilla_H3_aligned[i] and curr_close > ema_34_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Price breaks below L3 AND price < 1d EMA34 (downtrend)
                elif curr_close < camarilla_L3_aligned[i] and curr_close < ema_34_aligned[i]:
                    signals[i] = -0.25
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
                signals[i] = 0.25
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
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0