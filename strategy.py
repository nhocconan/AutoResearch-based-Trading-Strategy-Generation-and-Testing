#!/usr/bin/env python3
"""
Experiment #1910: 1d Donchian(20) Breakout + 1w Trend Filter + Volume Confirmation
HYPOTHESIS: Daily Donchian channel breakouts capture significant momentum moves. 
Weekly EMA(20) filter ensures alignment with higher-timeframe trend to avoid counter-trend whipsaws.
Volume confirmation (>1.5x 20-day average) filters false breakouts.
ATR-based stoploss (2.5x ATR(14)) manages risk.
Target: 30-100 total trades over 4 years (7-25/year) on BTC/ETH/SOL.
Works in bull markets via breakouts and bear markets via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1910_1d_donchian20_1w_trend_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_1w = np.where(close_1w > ema_20_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators ===
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-day average
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
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), EMA(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper band AND 1w trend up
            if trend_1w_aligned[i] > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower band AND 1w trend down
            elif trend_1w_aligned[i] < 0 and price < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals