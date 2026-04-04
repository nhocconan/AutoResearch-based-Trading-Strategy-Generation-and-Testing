#!/usr/bin/env python3
"""
Experiment #3793: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: 4h Donchian breakouts capture swing moves with 12h HMA(21) filtering counter-trend noise and volume (>1.5x MA) confirming institutional participation. Works in bull markets (breakouts above resistance with HMA up) and bear markets (breakdowns below support with HMA down). Discrete position sizing (0.25) minimizes fee drag. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3793_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h close
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        hma = 2 * wma_half - wma_full
        hma = pd.Series(hma).ewm(span=sqrt, adjust=False).mean().values
        return hma
    
    hma_12h = calculate_hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                # Calculate ATR(14) for exit condition
                if i >= 14:
                    prev_close = close[i-1]
                    tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
                    atr_14 = tr  # Simplified: use current TR as ATR proxy for exit
                    if price < highest_since_entry - 2.0 * atr_14:
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
                    signals[i] = SIZE  # Hold until enough data for ATR
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if i >= 14:
                    prev_close = close[i-1]
                    tr = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
                    atr_14 = tr  # Simplified: use current TR as ATR proxy for exit
                    if price > lowest_since_entry + 2.0 * atr_14:
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
                    signals[i] = -SIZE  # Hold until enough data for ATR
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) AND HMA trend alignment
        volume_spike = vol_ratio[i] > 1.5
        hma_uptrend = hma_12h_aligned[i] > hma_12h_aligned[i-1] if i > 0 else False
        hma_downtrend = hma_12h_aligned[i] < hma_12h_aligned[i-1] if i > 0 else False
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND HMA trending up
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                hma_uptrend):                  # HMA trending up (bullish bias)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND HMA trending down
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  hma_downtrend):                # HMA trending down (bearish bias)
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