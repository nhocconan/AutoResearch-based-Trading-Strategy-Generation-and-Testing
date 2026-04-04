#!/usr/bin/env python3
"""
Experiment #5597: 4h Donchian(20) breakout + 1d/1w HTF regime filter + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 1.7x average, filtered by 1d ADX (trending) and 1w pivot position (bull/bear bias), capture high-probability moves. Uses ATR(14) trailing stop (2.5x). Discrete position sizing (0.28) balances return and fee drag. Target: 19-50 trades/year (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5597_4h_donchian20_1d1w_regime_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for ADX (trend strength) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 30:
        # Calculate ADX(14) on 1d data
        plus_dm = np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0])
        minus_dm = np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0]) * -1
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        tr1 = df_1d['high'].values - df_1d['low'].values
        tr2 = np.abs(np.diff(df_1d['high'].values, prepend=df_1d['high'].values[0]))
        tr3 = np.abs(np.diff(df_1d['low'].values, prepend=df_1d['low'].values[0]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_1d > 0, atr_1d, 1)
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / np.where(atr_1d > 0, atr_1d, 1)
        dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) > 0, (plus_di + minus_di), 1)
        adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for Weekly Pivot (regime bias) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Calculate weekly pivot from prior week's OHLC
        weekly_high = pd.Series(df_1w['high'].values).shift(1)
        weekly_low = pd.Series(df_1w['low'].values).shift(1)
        weekly_close = pd.Series(df_1w['close'].values).shift(1)
        
        # Weekly Pivot Point (PP) = (H + L + C) / 3
        pp = (weekly_high + weekly_low + weekly_close) / 3.0
        # Weekly R1 = PP + (H - L)
        r1 = pp + (weekly_high - weekly_low)
        # Weekly S1 = PP - (H - L)
        s1 = pp - (weekly_high - weekly_low)
        
        # Align to LTF (4h)
        pp_aligned = align_htf_to_ltf(prices, df_1w, pp.values)
        r1_aligned = align_htf_to_ltf(prices, df_1w, r1.values)
        s1_aligned = align_htf_to_ltf(prices, df_1w, s1.values)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 30)  # Donchian, volume avg, ATR, ADX
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below S1 (mean reversion) OR ADX weak (<20) AND price < PP
                if price <= stop_price or price <= s1_aligned[i] or (adx_1d_aligned[i] < 20 and price < pp_aligned[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above R1 (mean reversion) OR ADX weak (<20) AND price > PP
                if price >= stop_price or price >= r1_aligned[i] or (adx_1d_aligned[i] < 20 and price > pp_aligned[i]):
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.7
        
        # Regime filter: ADX > 25 for trending market
        trending = adx_1d_aligned[i] > 25
        
        # Determine bias from Weekly Pivot levels
        # Bull bias: price > PP, Bear bias: price < PP
        bull_bias = price > pp_aligned[i]
        bear_bias = price < pp_aligned[i]
        
        # Long: breakout up + volume + trending + bull bias OR breakout up + volume + bull bias (mean reversion in range)
        long_setup = (breakout_up and volume_confirmed and 
                     ((trending and bull_bias) or (not trending and bull_bias and price < r1_aligned[i])))
        
        # Short: breakout down + volume + trending + bear bias OR breakout down + volume + bear bias (mean reversion in range)
        short_setup = (breakout_down and volume_confirmed and 
                      ((trending and bear_bias) or (not trending and bear_bias and price > s1_aligned[i])))
        
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