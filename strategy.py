#!/usr/bin/env python3
"""
Experiment #3682: 12h Donchian(20) breakout + 1d/1w HTF filters + volume confirmation
HYPOTHESIS: 12h Donchian breakouts capture intermediate-term swings while 1d EMA50 and 1w supertrend provide multi-timeframe trend alignment. Volume spike confirms breakout authenticity. This avoids counter-trend trades and works in both bull (breakouts with volume above EMA50 and supertrend up) and bear (failed breakouts below EMA50 and supertrend down reverse quickly) markets. Position size 0.25 balances return and drawdown. Target: 75-150 total trades over 4 years (19-37/year) by using strict entry conditions requiring Donchian breakout, multi-timeframe trend alignment, and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3682_12h_donchian20_1d_1w_filters_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === HTF: 1w data for Supertrend trend filter ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1_w = high_1w[1:] - low_1w[1:]
    tr2_w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    atr_w = pd.Series(tr_w).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2_w = (high_1w + low_1w) / 2.0
    upper_w = hl2_w + 3.0 * atr_w
    lower_w = hl2_w - 3.0 * atr_w
    
    supertrend_w = np.full(len(close_1w), np.nan, dtype=np.float64)
    dir_w = np.full(len(close_1w), 1, dtype=np.int8)  # 1 for up, -1 for down
    
    for i in range(1, len(close_1w)):
        if close_1w[i] > upper_w[i-1]:
            dir_w[i] = 1
        elif close_1w[i] < lower_w[i-1]:
            dir_w[i] = -1
        else:
            dir_w[i] = dir_w[i-1]
            if dir_w[i] == 1 and lower_w[i] < lower_w[i-1]:
                lower_w[i] = lower_w[i-1]
            if dir_w[i] == -1 and upper_w[i] > upper_w[i-1]:
                upper_w[i] = upper_w[i-1]
        
        if dir_w[i] == 1:
            supertrend_w[i] = lower_w[i]
        else:
            supertrend_w[i] = upper_w[i]
    
    supertrend_w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_w)
    
    # === 12h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_dc + 1, 20, 14, 50, 10)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(supertrend_w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below 1d EMA50 (trend reversal)
                elif price < ema50_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if 1w supertrend turns down
                elif supertrend_w_aligned[i] > price:  # price below supertrend = downtrend
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above 1d EMA50 (trend reversal)
                elif price > ema50_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if 1w supertrend turns up
                elif supertrend_w_aligned[i] < price:  # price above supertrend = uptrend
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND 
            # price above 1d EMA50 AND price above 1w supertrend (bullish alignment)
            if (price > highest_high[i-1] and   # Breakout above previous period's high
                price > ema50_1d_aligned[i] and   # Above 1d EMA50 (bullish bias)
                price > supertrend_w_aligned[i]): # Above 1w supertrend (bullish bias)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND 
            # price below 1d EMA50 AND price below 1w supertrend (bearish alignment)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < ema50_1d_aligned[i] and  # Below 1d EMA50 (bearish bias)
                  price < supertrend_w_aligned[i]): # Below 1w supertrend (bearish bias)
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals