#!/usr/bin/env python3
"""
Experiment #3834: 1h Donchian(20) breakout + 4h EMA(50) trend filter + 1d volume confirmation
HYPOTHESIS: 1h Donchian breakouts capture swing moves with 4h EMA(50) filtering for trend direction (bullish above EMA, bearish below) and 1d volume (>1.5x) confirming institutional participation. Uses 1h timeframe for precise entry timing while 4h/1d provide signal direction. Session filter (08-20 UTC) reduces noise. Discrete position sizing (0.20) minimizes fee drag. Target: 60-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3834_1h_donchian20_4h_ema50_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for EMA(50) trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(volume_1d))
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 50, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Session Filter ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price < highest_since_entry - 2.5 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price breaks below Donchian lower band (trend reversal)
                    elif price < lowest_low[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = SIZE
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if i > 0:
                    atr_temp = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
                    if price > lowest_since_entry + 2.5 * atr_temp:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    # Exit if price breaks above Donchian upper band (trend reversal)
                    elif price > highest_high[i]:
                        in_position = False
                        position_side = 0
                        signals[i] = 0.0
                    else:
                        signals[i] = -SIZE
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d volume spike (> 1.5x average)
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above 4h EMA(50) (bullish trend)
            if (price > highest_high[i-1] and    # Breakout above previous period's high
                price > ema_4h_aligned[i]):      # Above 4h EMA(50) (bullish trend filter)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below 4h EMA(50) (bearish trend)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < ema_4h_aligned[i]):    # Below 4h EMA(50) (bearish trend filter)
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