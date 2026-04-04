#!/usr/bin/env python3
"""
Experiment #4122: 12h Donchian(20) breakout + 1d EMA50 filter + volume confirmation + chop regime
HYPOTHESIS: 12h Donchian breakouts aligned with daily EMA50 trend and volume confirmation capture institutional order flow. Adding choppiness regime filter (CHOP > 61.8) avoids whipsaws in ranging markets. Works in bull/bear by using daily EMA as trend filter and chop filter to avoid false signals. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4122_12h_donchian20_1d_ema_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        close_1d = df_1d['close'].values
        ema_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    else:
        ema_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1d Chopiness Index(14) for regime filter ===
    if len(df_1d) >= 1:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # ATR(14)
        atr_1d = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Highest high and lowest low over 14 periods
        highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Chopiness Index: 100 * log10(sum(ATR14) / (max(HH14) - min(LL14))) / log10(14)
        sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
        hh_ll_diff = highest_high_14 - lowest_low_14
        chop_1d = np.where(
            (hh_ll_diff > 0) & (~np.isnan(sum_atr_14)) & (~np.isnan(hh_ll_diff)),
            100 * np.log10(sum_atr_14 / hh_ll_diff) / np.log10(14),
            50.0  # neutral when invalid
        )
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        chop_1d_aligned = np.full(n, 50.0)
    
    # === 12h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 12h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(20) for volatility and stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20 + 10, 50 + 10, 14 + 10)  # DC, vol MA, EMA, chop buffers
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
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
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        # Require choppy market (CHOP > 61.8) for mean reversion avoidance - actually we want TRENDING
        # Chop > 61.8 = ranging, Chop < 38.2 = trending
        # We want trending markets for breakouts, so chop < 38.2
        trending_regime = chop_1d_aligned[i] < 38.2
        
        if volume_spike and trending_regime:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Daily EMA trend filter
            price_above_daily_ema = price > ema_1d_aligned[i]
            price_below_daily_ema = price < ema_1d_aligned[i]
            
            # Long conditions: Donchian breakout up + price above daily EMA
            long_entry = breakout_up and price_above_daily_ema
            
            # Short conditions: Donchian breakout down + price below daily EMA
            short_entry = breakout_down and price_below_daily_ema
            
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
        else:
            signals[i] = 0.0
    
    return signals