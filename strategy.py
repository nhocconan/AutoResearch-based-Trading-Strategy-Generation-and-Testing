#!/usr/bin/env python3
"""
Experiment #4354: 1h Donchian(20) + 4h EMA(200) + 1d Volume Spike + Session Filter
HYPOTHESIS: 1h Donchian breakouts in direction of 4h EMA(200) trend, confirmed by 1d volume > 2.0x average, during active session (08-20 UTC). Uses 4h for trend filter, 1d for volume regime, 1h for precise entry timing. Targets 60-150 total trades over 4 years (15-37/year) with position size 0.20. Works in bull via buying breakouts above EMA200, in bear via selling breakdowns below EMA200. Volume spike filter ensures institutional participation, reducing false breakouts. Session filter avoids low-liquidity Asian session noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4354_1h_donchian20_4h_ema200_1d_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 4h EMA(200) for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) >= 200:
        ema_200_4h = pd.Series(df_4h['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    else:
        ema_200_4h_aligned = np.full(n, np.nan)
    
    # === Precompute HTF: 1d Volume MA(20) for volume regime filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
        vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    else:
        vol_ma_1d_aligned = np.full(n, np.nan)
    
    # === 1h Indicators: Donchian Channel (20) ===
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr[i]) or
            np.isnan(ema_200_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
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
        # Volume confirmation: current 1h volume > 2.0x 1d average volume (scaled)
        # Approximate 1d volume from aligned 1d MA (represents typical daily volume)
        vol_confirm = volume[i] > (vol_ma_1d_aligned[i] * 2.0 / 24.0)  # 1d MA / 24 = approx hourly average
        
        # Trend filter: price relative to 4h EMA200
        above_ema = price > ema_200_4h_aligned[i]
        below_ema = price < ema_200_4h_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = price > donch_high[i-1]  # Break above previous period's high
        breakout_down = price < donch_low[i-1]  # Break below previous period's low
        
        if vol_confirm:
            # Long: breakout up + price above 4h EMA200 (uptrend)
            if breakout_up and above_ema:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short: breakout down + price below 4h EMA200 (downtrend)
            elif breakout_down and below_ema:
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