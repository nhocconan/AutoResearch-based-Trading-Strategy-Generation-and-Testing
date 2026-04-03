#!/usr/bin/env python3
"""
Experiment #368: 12h Donchian Breakout + Weekly Trend Filter + Volume Spike

HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts capture significant momentum moves.
Weekly trend (price vs 50-period EMA) filters for higher-probability direction.
Volume spike (>1.8x 20-period average) confirms institutional participation.
ATR(14) stoploss manages risk. Designed for low trade frequency (target 15-25/year)
to minimize fee drag while capturing large trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_weekly_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA(50) for trend
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50 = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        weekly_trend_up = np.zeros(len(close_1w), dtype=bool)
        weekly_trend_down = np.zeros(len(close_1w), dtype=bool)
        weekly_trend_up[50:] = close_1w[50:] > ema_50[50:]
        weekly_trend_down[50:] = close_1w[50:] < ema_50[50:]
        weekly_trend_up[:50] = False
        weekly_trend_down[:50] = False
        weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
        weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    else:
        weekly_trend_up_aligned = np.full(n, False)
        weekly_trend_down_aligned = np.full(n, False)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === LTF: 12h Donchian channels (20-period) ===
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        highest_high[i] = np.max(high[i - lookback + 1:i + 1])
        lowest_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, lookback)  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian lower band (trailing stop)
                if close[i] <= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian upper band (trailing stop)
                if close[i] >= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian upper band + weekly uptrend + volume spike
        long_condition = (
            close[i] > highest_high[i] and
            weekly_trend_up_aligned[i] and
            vol_ratio_1d_aligned[i] > 1.8
        )
        
        # Short: Break below Donchian lower band + weekly downtrend + volume spike
        short_condition = (
            close[i] < lowest_low[i] and
            weekly_trend_down_aligned[i] and
            vol_ratio_1d_aligned[i] > 1.8
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals