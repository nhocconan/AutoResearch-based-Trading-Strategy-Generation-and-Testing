#!/usr/bin/env python3
"""
Experiment #4529: 4h Donchian(20) Breakout + 1d/1w EMA Trend + Volume Confirmation
HYPOTHESIS: 4h Donchian(20) breakouts aligned with 1d/1w EMA trend (bullish when price > both EMAs, bearish when price < both) and volume confirmation (>1.5x average) capture medium-term momentum with higher timeframe trend filter. This avoids counter-trend trades in strong trends, reducing whipsaw in ranging/choppy markets. Designed for 4h timeframe to target 75-200 total trades over 4 years (19-50/year) with position size 0.25. Works in both bull and bear markets by only trading breakouts in direction of higher timeframe EMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4529_4h_donchian20_1d_1w_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d and 1w data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) for 1d and 1w
    if len(df_1d) >= 50:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_1d = np.array([])
        
    if len(df_1w) >= 50:
        ema_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_1w = np.array([])
    
    # Align to 4h timeframe
    if len(ema_1d) > 0:
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
        
    if len(ema_1w) > 0:
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50)  # Donchian, vol MA, ATR, EMA warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
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
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Higher timeframe trend filter: bullish when price > both EMAs, bearish when price < both
        htf_bullish = price > ema_1d_aligned[i] and price > ema_1w_aligned[i]
        htf_bearish = price < ema_1d_aligned[i] and price < ema_1w_aligned[i]
        
        # Donchian breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout in bullish HTF trend + volume
        long_entry = breakout_up and htf_bullish and volume_confirm
        
        # Short conditions: downward breakout in bearish HTF trend + volume
        short_entry = breakout_down and htf_bearish and volume_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals