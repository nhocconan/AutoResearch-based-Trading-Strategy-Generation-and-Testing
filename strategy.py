#!/usr/bin/env python3
"""
Experiment #4309: 4h Donchian(20) breakout + 1d HMA(50) trend + volume confirmation + chop filter
HYPOTHESIS: Donchian breakouts on 4h capture swing momentum when aligned with 1d HMA50 trend, confirmed by volume (>2.0x average), and filtered by choppiness regime (CHOP>61.8 for mean reversion in chop, CHOP<38.2 for trend following). Uses discrete position sizing (0.25) to minimize fee churn. ATR trailing stop (2.5x) manages risk. Targets 75-200 total trades over 4 years by requiring confluence of breakout, trend, volume, and regime filters. Works in bull via breakout continuation, in bear via shorting breakdowns with regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4309_4h_donchian20_1d_hma_vol_chop_v1"
timeframe = "4h"
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
    
    # === Precompute HTF: 1d HMA50 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 50:
        # Calculate HMA(50): WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = 50 // 2
        sqrt_n = int(np.sqrt(50))
        wma_half = pd.Series(df_1d['close'].values).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_1d = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === Precompute HTF: 1w Chopiness Index for regime filter ===
    df_1w = get_htf_data(prices, '1w')
    chop_1w = np.full(n, 50.0)  # default to neutral chop
    if len(df_1w) >= 14:
        tr1 = df_1w['high'].values[1:] - df_1w['low'].values[1:]
        tr2 = np.abs(df_1w['high'].values[1:] - df_1w['close'].values[:-1])
        tr3 = np.abs(df_1w['low'].values[1:] - df_1w['close'].values[:-1])
        tr_1w = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
        atr_1w = pd.Series(tr_1w).ewm(span=14, min_periods=14, adjust=False).mean().values
        sum_tr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        highest_h_1w = pd.Series(df_1w['high'].values).rolling(window=14, min_periods=14).max().values
        lowest_l_1w = pd.Series(df_1w['low'].values).rolling(window=14, min_periods=14).min().values
        chop_1w = 100 * np.log10(sum_tr_1w / (highest_h_1w - lowest_l_1w)) / np.log10(14)
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, 50.0)
    
    # === 4h Indicators: Donchian Channel (20) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_upper, donch_lower = calculate_donchian(high, low, 20)
    
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
    
    warmup = max(20, 20, 14, 50, 14)  # Donchian, vol MA, ATR, 1d HMA, 1w Chop
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(chop_1w_aligned[i])):
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
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        if volume_confirm:
            # Donchian breakout conditions (using previous bar's levels)
            breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
            breakout_dn = close[i] < donch_lower[i-1]  # Close below previous lower band
            
            # 1d HMA50 trend filter
            price_above_hma = price > hma_1d_aligned[i]
            price_below_hma = price < hma_1d_aligned[i]
            
            # 1w Chopiness regime filter
            chop_value = chop_1w_aligned[i]
            chop_trending = chop_value < 38.2   # Trending regime (follow breakout)
            chop_chopping = chop_value > 61.8   # Chopping regime (mean reversion)
            
            # Long conditions: Donchian breakout up + price above HMA50 + regime filter
            long_entry = breakout_up and price_above_hma and (chop_trending or chop_chopping)
            
            # Short conditions: Donchian breakout down + price below HMA50 + regime filter
            short_entry = breakout_dn and price_below_hma and (chop_trending or chop_chopping)
            
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