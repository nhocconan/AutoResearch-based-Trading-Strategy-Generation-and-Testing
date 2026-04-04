#!/usr/bin/env python3
"""
Experiment #5014: 1h Donchian(20) Breakout + 4h/1d EMA Trend + Volume Spike + ATR Stoploss
HYPOTHESIS: On 1h timeframe, Donchian(20) breakouts aligned with 4h/1d EMA trend (EMA50 > EMA200 for long, EMA50 < EMA200 for short) with volume confirmation (>1.5x average) capture momentum moves. Uses ATR(14) trailing stop (2.0x) to limit downside. Designed for 15-37 trades/year on 1h timeframe (60-150 total over 4 years) to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend). Session filter (08-20 UTC) reduces noise trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5014_1h_donchian20_4h_1d_ema_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 4h and 1d data for EMA trend filters
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h Indicators: EMA50 and EMA200 for trend filter ===
    if len(df_4h) >= 200:
        close_4h = df_4h['close'].values
        ema50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema200_4h = pd.Series(close_4h).ewm(span=200, min_periods=200, adjust=False).mean().values
        # Trend: 1 = bullish (EMA50 > EMA200), -1 = bearish (EMA50 < EMA200), 0 = neutral
        trend_4h = np.where(ema50_4h > ema200_4h, 1, np.where(ema50_4h < ema200_4h, -1, 0))
    else:
        trend_4h = np.zeros(len(df_4h))
    
    # Align 4h trend to 1h timeframe
    if len(trend_4h) > 0:
        trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    else:
        trend_4h_aligned = np.full(n, 0)
    
    # === 1d Indicators: EMA50 and EMA200 for trend filter ===
    if len(df_1d) >= 200:
        close_1d = df_1d['close'].values
        ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
        # Trend: 1 = bullish (EMA50 > EMA200), -1 = bearish (EMA50 < EMA200), 0 = neutral
        trend_1d = np.where(ema50_1d > ema200_1d, 1, np.where(ema50_1d < ema200_1d, -1, 0))
    else:
        trend_1d = np.zeros(len(df_1d))
    
    # Align 1d trend to 1h timeframe
    if len(trend_1d) > 0:
        trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    else:
        trend_1d_aligned = np.full(n, 0)
    
    # === 1h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: Volume confirmation (1.5x spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour  # open_time is already datetime64[ms], index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Require both 4h and 1d trend to align (both bullish or both bearish)
        trend_bullish = (trend_4h_aligned[i] > 0) and (trend_1d_aligned[i] > 0)
        trend_bearish = (trend_4h_aligned[i] < 0) and (trend_1d_aligned[i] < 0)
        
        # Donchian breakout conditions with trend alignment
        breakout_long = (price >= high_roll[i]) and trend_bullish and vol_confirm
        breakout_short = (price <= low_roll[i]) and trend_bearish and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</think>