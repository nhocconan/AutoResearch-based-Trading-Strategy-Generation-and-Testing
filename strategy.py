#!/usr/bin/env python3
"""
Experiment #5725: 12h Donchian(20) breakout + 1d EMA(200) trend + volume confirmation + chop filter
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned 
with daily EMA(200) direction (price above EMA = bullish, below = bearish) capture 
high-probability trend continuation moves. The daily EMA(200) provides a robust long-term 
trend filter that adapts to both bull and bear markets. Volume confirms breakout strength. 
Choppiness Index (CHOP) > 61.8 avoids ranging markets. ATR trailing stop (2.5x) manages risk. 
Discrete sizing (0.25) minimizes fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5725_12h_donchian20_1d_ema200_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for EMA(200) and Chop filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 200:
        ema_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    else:
        ema_1d = np.full(len(df_1d), np.nan)
    
    # Choppiness Index on 1d (14-period)
    if len(df_1d) >= 14:
        tr1 = df_1d['high'].values - df_1d['low'].values
        tr2 = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
        tr3 = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        hh = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
        sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        denominator = hh - ll
        chop = np.where(denominator > 0, 100 * np.log10(sum_tr / denominator) / np.log10(14), 50)
    else:
        atr_1d = np.full(len(df_1d), np.nan)
        chop = np.full(len(df_1d), 50.0)
    
    # Align daily indicators to 12h timeframe (shifted by 1 for completed daily bars only)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 12h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 200)  # Donchian, volume avg, ATR, EMA
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below daily EMA (trend change) OR chop > 61.8 (range)
                if price <= stop_price or price <= ema_1d_aligned[i] or chop_aligned[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above daily EMA (trend change) OR chop > 61.8 (range)
                if price >= stop_price or price >= ema_1d_aligned[i] or chop_aligned[i] > 61.8:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8
        trending_market = chop_aligned[i] <= 61.8  # Only trade in trending markets
        
        # Daily EMA bias: long above EMA, short below EMA
        long_bias = price > ema_1d_aligned[i]
        short_bias = price < ema_1d_aligned[i]
        
        # Entry conditions: breakout in direction of daily EMA with volume and trend
        long_setup = breakout_up and volume_confirmed and long_bias and trending_market
        short_setup = breakout_down and volume_confirmed and short_bias and trending_market
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals