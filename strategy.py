#!/usr/bin/env python3
"""
Experiment #3650: 1d Donchian(20) + 1w HMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: Daily Donchian breakouts capture medium-term momentum while weekly HMA filters for higher-timeframe trend alignment. Volume spike confirms breakout strength. ATR-based trailing stop manages risk. Designed for 1d timeframe to minimize fee drag (target 30-100 trades over 4 years). Works in bull markets (breakouts with trend) and bear markets (fade false breakouts against trend by requiring trend alignment).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3650_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate weekly HMA(21) for trend direction
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    if half_len > 0 and sqrt_len > 0:
        wma_half = pd.Series(close_1w).rolling(window=half_len, min_periods=half_len).mean().values
        wma_full = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_1w = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
    else:
        hma_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # === 1d Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for volatility and stoploss ===
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
    
    warmup = max(50, lookback_dc + 1, 21, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
                # Exit if Donchian breakout fails (price re-enters channel)
                elif price < highest_high[i-1]:  # Note: i-1 to avoid look-ahead
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
                # Exit if Donchian breakout fails (price re-enters channel)
                elif price > lowest_low[i-1]:  # Note: i-1 to avoid look-ahead
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
            # Determine trend bias from weekly HMA
            bullish_bias = hma_1w_aligned[i] > close_1w[-1] if len(close_1w) > 0 else price > hma_1w_aligned[i]
            bearish_bias = hma_1w_aligned[i] < close_1w[-1] if len(close_1w) > 0 else price < hma_1w_aligned[i]
            
            # Long entry: Price breaks above Donchian upper band in bullish trend
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                bullish_bias):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band in bearish trend
            elif (price < lowest_low[i-1] and   # Breakout below previous period's low
                  bearish_bias):
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