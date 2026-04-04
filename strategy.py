#!/usr/bin/env python3
"""
Experiment #5337: 4h Donchian(20) breakout + 1d EMA trend + volume confirmation + chop filter
HYPOTHESIS: On 4h timeframe, price breaking above/below the 20-period Donchian channel 
with volume > 1.5x average, aligned with 1d EMA(50) trend, and in non-choppy market 
(CHOP > 61.8 = trending) captures strong momentum moves. Uses discrete position sizing 
(0.25) and ATR-based trailing stoploss. Target: 19-50 trades/year on 4h timeframe 
(75-200 total over 4 years) to minimize fee drag. Works in bull markets via breakouts 
and in bear markets via short breakdowns with trend + regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5337_4h_donchian20_1d_ema_vol_chop_v1"
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
    
    # === HTF: 1d data for EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    else:
        ema_50 = np.array([])
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50) if len(ema_50) > 0 else np.full(n, np.nan)
    
    # === HTF: 1d data for Choppiness Index regime filter ===
    if len(df_1d) >= 14:
        # True Range for 1d
        tr1_1d = df_1d['high'].values - df_1d['low'].values
        tr2_1d = np.abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1))
        tr3_1d = np.abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))
        tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
        tr_1d[0] = tr1_1d[0]
        atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
        
        # Highest high and lowest low over 14 periods
        hh_1d = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
        ll_1d = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
        sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        hh_ll_diff = hh_1d - ll_1d
        chop = np.where(hh_ll_diff > 0, 100 * np.log10(sum_atr_14 / hh_ll_diff) / np.log10(14), 100)
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop) if len(chop) > 0 else np.full(n, 50.0)
    else:
        chop_aligned = np.full(n, 50.0)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 50, 14)  # Donchian, volume avg, ATR, EMA50, CHOP warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: Only trade in trending markets (CHOP > 61.8) ---
        is_trending = chop_aligned[i] > 61.8
        
        if not is_trending:
            # In choppy markets, stay flat
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # --- Exit Logic: Close position on stoploss or trend reversal ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                if price <= stop_price or price <= donchian_low[i] or price < ema_50_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                if price >= stop_price or price >= donchian_high[i] or price > ema_50_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        ema_trend_up = price > ema_50_aligned[i-1]
        ema_trend_down = price < ema_50_aligned[i-1]
        
        if breakout_up and volume_confirmed and ema_trend_up:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_down and volume_confirmed and ema_trend_down:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals