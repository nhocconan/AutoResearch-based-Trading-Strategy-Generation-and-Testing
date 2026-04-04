#!/usr/bin/env python3
"""
Experiment #3788: 12h Donchian(20) breakout + 1w/1d HTF trend filter + volume confirmation
HYPOTHESIS: 12h Donchian breakouts capture medium-term swings. HTF trend (1w EMA50 > 1d EMA20) filters for institutional alignment. Volume spike (>1.5x) confirms participation. Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend). Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3788_12h_donchian20_1w_1d_trend_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for EMA50 trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate 1w EMA(50)
    ema_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === HTF: 1d data for EMA20 trend and volume profile ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    # Calculate 1d EMA(20)
    ema_1d = pd.Series(close_1d).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    # Calculate 1d volume profile high-volume node (VHN)
    nbins = 30
    vhn_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 1:
            continue
        hist, bin_edges = np.histogram(
            [high_1d[i], low_1d[i], close_1d[i]],
            bins=nbins,
            range=(low_1d[i], high_1d[i]),
            weights=[volume_1d[i], volume_1d[i], volume_1d[i]]
        )
        if np.sum(hist) > 0:
            max_bin_idx = np.argmax(hist)
            vhn_1d[i] = (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1]) / 2
    vhn_1d_aligned = align_htf_to_ltf(prices, df_1d, vhn_1d)
    
    # === 12h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = max(lookback_dc + 1, 50, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(vhn_1d_aligned[i]) or np.isnan(vol_ratio[i])):
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
                    tr = np.maximum(high[i] - low[i], np.maximum(np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1])))
                    atr_14 = pd.Series([tr if j==i else np.nan for j in range(i+1)]).rolling(window=14, min_periods=1).mean().iloc[-1]
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
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if i >= 14:
                    tr = np.maximum(high[i] - low[i], np.maximum(np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1])))
                    atr_14 = pd.Series([tr if j==i else np.nan for j in range(i+1)]).rolling(window=14, min_periods=1).mean().iloc[-1]
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
            continue
        
        # --- New Position Entry Logic ---
        # HTF trend: 1w EMA50 > 1d EMA20 for uptrend, < for downtrend
        uptrend = ema_1w_aligned[i] > ema_1d_aligned[i]
        downtrend = ema_1w_aligned[i] < ema_1d_aligned[i]
        # Volume confirmation
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND HTF uptrend
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                uptrend):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND HTF downtrend
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  downtrend):
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