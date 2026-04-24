#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R mean reversion with 1d EMA200 trend filter and volume spike confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA200 for trend filter (price > EMA200 = uptrend bias, price < EMA200 = downtrend bias).
- Williams %R(14) for mean reversion: Long when %R < -80 (oversold) in uptrend, Short when %R > -20 (overbought) in downtrend.
- Volume confirmation: volume > 2.0 * 4h volume MA(20) to ensure conviction on reversals.
- Exit: ATR-based trailing stop (2.5 * ATR(14)) from highest high/lowest low since entry.
- Signal size: 0.30 discrete to control fee drag.
- Designed to work in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets with controlled drawdown.
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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate Williams %R(14) for 4h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate ATR(14) for 4h timeframe
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[high[0] - low[0]], tr])  # first TR is high-low
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume MA(20) for 4h timeframe
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start from index where all indicators are ready
    start_idx = max(200, 14, 20)  # EMA200 needs 200, Williams %R needs 14, volume MA needs 20, ATR needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(williams_r[i]) or 
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
                # Long: Williams %R < -80 (oversold) AND price > 1d EMA200 (uptrend bias)
                if williams_r[i] < -80 and curr_close > ema_200_aligned[i]:
                    signals[i] = 0.30
                    position = 1
                    entry_price = curr_close
                    highest_since_entry = curr_close
                    lowest_since_entry = curr_close
                # Short: Williams %R > -20 (overbought) AND price < 1d EMA200 (downtrend bias)
                elif williams_r[i] > -20 and curr_close < ema_200_aligned[i]:
                    signals[i] = -0.30
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
                signals[i] = 0.30
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
                signals[i] = -0.30
    
    return signals

name = "4h_WilliamsR_MeanReversion_1dEMA200_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0