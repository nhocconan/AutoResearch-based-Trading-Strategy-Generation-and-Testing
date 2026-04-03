#!/usr/bin/env python3
"""
Experiment #1983: 4h Donchian(20) breakout + 12h trend + volume confirmation
HYPOTHESIS: Donchian channel breakouts on 4h timeframe capture institutional momentum. 
Filter with 12h EMA(50) trend for alignment and volume spike (>1.5x 20-period average) for confirmation.
ATR-based stoploss and discrete position sizing (0.25) minimize fee drag.
Target: 75-200 total trades over 4 years (19-50/year) with Sharpe > 0 on all symbols.
Works in bull/bear markets by following higher timeframe trend with volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1983_4h_donchian20_12h_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_12h = np.where(close_12h > ema_50_12h, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 4h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20), EMA(50), ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: stoploss or Donchian opposite touch
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2 * ATR below entry
                if price < entry_price - 2.0 * atr[i]:
                    exit_signal = True
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= lowest_20[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2 * ATR above entry
                if price > entry_price + 2.0 * atr[i]:
                    exit_signal = True
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= highest_20[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h trend alignment for bias filter
        trend_bias = trend_12h_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND 12h trend up
            if trend_bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND 12h trend down
            elif trend_bias < 0 and price < lowest_20[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals
}