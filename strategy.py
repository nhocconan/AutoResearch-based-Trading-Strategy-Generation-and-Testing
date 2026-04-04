#!/usr/bin/env python3
"""
Experiment #3667: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture medium-term momentum while 1d weekly pivot levels (from prior week) provide institutional reference points for breakout/fade decisions. Volume spike confirms breakout authenticity. Weekly pivot bias filters for higher-timeframe structure: long only when price > weekly pivot, short only when price < weekly pivot. This avoids counter-trend trades against weekly momentum. Position size 0.25 balances return and drawdown. Target: 75-150 total trades over 4 years (19-37/year) by using strict entry conditions requiring both Donchian breakout and pivot alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3667_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate weekly pivot points from prior week's OHLC
    # Need to group daily data into weeks (Mon-Sun) and calculate pivot for prior week
    # We'll compute weekly OHLC then derive pivot: (H+L+C)/3
    # Since we don't have explicit week grouping, approximate using rolling window of 5 days
    # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
    # Use 5-day rolling for weekly approximation (trading week)
    lookback_week = 5
    weekly_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().values
    weekly_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().values
    weekly_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).mean().values  # approx weekly close
    
    # Weekly pivot point: (H+L+C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and stoploss ===
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
    
    warmup = max(50, lookback_dc + 1, lookback_week + 1, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                # Exit if price breaks below weekly pivot (trend reversal vs weekly bias)
                elif price < weekly_pivot_aligned[i]:
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
                # Exit if price breaks above weekly pivot (trend reversal vs weekly bias)
                elif price > weekly_pivot_aligned[i]:
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
            # Long entry: Price breaks above Donchian upper band AND above weekly pivot (bullish alignment)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > weekly_pivot_aligned[i]):   # Above weekly pivot (bullish weekly bias)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below weekly pivot (bearish alignment)
            elif (price < lowest_low[i-1] and   # Breakout below previous period's low
                  price < weekly_pivot_aligned[i]):  # Below weekly pivot (bearish weekly bias)
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