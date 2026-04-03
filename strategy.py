#!/usr/bin/env python3
"""
Experiment #111: 6h Williams %R(14) + 1d EMA(21) trend + volume confirmation
HYPOTHESIS: On 6h timeframe, Williams %R identifies overbought/oversold conditions. 
In strong trends (price > 1d EMA), we fade extreme %R readings for mean reversion.
In weak trends (price < 1d EMA), we breakout in trend direction. 
Volume confirmation (>1.3x average) filters false signals. 
Target: 75-150 total trades over 4 years (19-38/year). Works in bull/bear via trend-adaptive logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_111_6h_williamsr_1d_ema_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA(21) trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = pd.Series(df_1d['close'].values)
    ema_1d = close_1d.ewm(span=21, adjust=False, min_periods=21).mean().values
    daily_trend = np.where(close_1d > ema_1d, 1, -1)  # 1 = bullish, -1 = bearish
    daily_trend_aligned = align_htf_to_ltf(prices, df_1d, daily_trend)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(daily_trend_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.3x average) ---
        volume_spike = vol_ratio[i] > 1.3
        
        # --- Williams %R Levels ---
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # --- Daily EMA Trend: from 1d data ---
        bullish_trend = daily_trend_aligned[i] > 0
        bearish_trend = daily_trend_aligned[i] < 0
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 12 bars (~3 days on 6h) to avoid overtrading
            if bars_since_entry > 12:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # In bullish trend: fade extreme oversold (mean reversion)
            if bullish_trend and oversold:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # In bearish trend: fade extreme overbought (mean reversion)
            elif bearish_trend and overbought:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            # In weak/transitioning trend: breakout in direction of %R momentum
            elif williams_r[i] > -50 and not oversold and not overbought:
                # Momentum long when %R rising from oversold
                if williams_r[i] > williams_r[i-1] and williams_r[i-1] < -80:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
            elif williams_r[i] < -50 and not oversold and not overbought:
                # Momentum short when %R falling from overbought
                if williams_r[i] < williams_r[i-1] and williams_r[i-1] > -20:
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