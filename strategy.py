#!/usr/bin/env python3
"""
Experiment #554: 1h Donchian(20) breakout + 4h EMA20 trend + 1d volume confirmation + ATR stoploss
HYPOTHESIS: 1h timeframe with strict multi-timeframe filters to minimize trades while capturing momentum.
Use 4h EMA20 for intermediate trend direction, 1d volume spike for participation confirmation, and 1h Donchian breakout for entry timing.
Session filter (08-20 UTC) reduces noise. Discrete position sizing (0.20) limits drawdown.
Target: 60-150 total trades over 4 years by requiring confluence of 3 filters + session.
Works in bull/bear: EMA20 trend filter avoids counter-trend trades, volume confirmation ensures legitimacy.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_554_1h_donchian20_4h_ema20_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for EMA20 trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA20 on 4h timeframe
    if len(close_4h) >= 20:
        ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    else:
        ema_4h = np.full(len(close_4h), np.nan)
    
    # Align EMA20 to 1h timeframe
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # === HTF: 1d data for volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period volume MA on daily timeframe
    if len(volume_1d) >= 20:
        vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    else:
        vol_ma_1d = np.full(len(volume_1d), np.nan)
    
    # Align 1d volume MA to 1h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # === 1h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA20 and Donchian
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        # --- Data Validity Check ---
        if (not in_session or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require 1h volume > 1.5x aligned 1d volume MA ---
        # Note: comparing 1h volume bar to 1d MA requires scaling - use 1/24th of daily MA as approximate hourly baseline
        vol_baseline = vol_ma_1d_aligned[i] / 24.0
        volume_spike = volume[i] > (1.5 * vol_baseline) if vol_baseline > 0 else False
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- 4h EMA20 Trend Filter ---
        bullish_trend = price > ema_4h_aligned[i]
        bearish_trend = price < ema_4h_aligned[i]
        
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
            
            # Optional: time-based exit after 24 bars (~1 day on 1h) to avoid overtrading
            if bars_since_entry > 24:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + bullish 4h EMA20 trend
            if breakout_up and bullish_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + bearish 4h EMA20 trend
            elif breakout_down and bearish_trend:
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