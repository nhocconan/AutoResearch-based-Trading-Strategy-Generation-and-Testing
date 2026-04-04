#!/usr/bin/env python3
"""
Experiment #3703: 4h Donchian(20) breakout + 12h/1d HMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts capture intermediate-term momentum with 12h/1d HMA confirming higher-timeframe trend alignment (bullish when price > HMA, bearish when price < HMA). Volume spike (>2.0x 20-bar average) validates breakout strength. Uses discrete position sizing (0.25) to limit drawdown from 2022 crash while allowing profit accumulation. Targets 75-200 trades over 4 years (19-50/year) with strict 3-condition confluence. ATR-based trailing stop (2.5x) manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3703_4h_donchian20_12h_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h and 1d data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) for 12h and 1d
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
        return hma_vals.values
    
    hma_12h = hma(df_12h['close'].values, 21)
    hma_1d = hma(df_1d['close'].values, 21)
    
    # Align HTF HMA to 4h timeframe (shifted by 1 for completed HTF bar)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(lookback_dc + 1, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]) or
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
                # Exit if price breaks below both HTF HMAs (trend change)
                elif price < hma_12h_aligned[i] and price < hma_1d_aligned[i]:
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
                # Exit if price breaks above both HTF HMAs (trend change)
                elif price > hma_12h_aligned[i] and price > hma_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above both HTF HMAs (bullish trend alignment)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > hma_12h_aligned[i] and   # Above 12h HMA (bullish bias)
                price > hma_1d_aligned[i]):      # Above 1d HMA (bullish bias)
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below both HTF HMAs (bearish trend alignment)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < hma_12h_aligned[i] and   # Below 12h HMA (bearish bias)
                  price < hma_1d_aligned[i]):      # Below 1d HMA (bearish bias)
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