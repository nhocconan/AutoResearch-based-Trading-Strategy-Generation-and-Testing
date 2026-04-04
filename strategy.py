#!/usr/bin/env python3
"""
Experiment #3543: 4h Donchian Breakout + 12h/1d Regime + Volume Spike
HYPOTHESIS: 4h Donchian(20) breakouts with 12h/1d regime filter (choppiness index) and volume confirmation capture medium-term momentum while avoiding false breakouts in ranging markets. The 12h timeframe provides intermediate trend direction, and 1d choppiness filter ensures we only trade in trending regimes. Position size 0.25. Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3543_4h_donchian20_12h_1d_chop_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Donchian channels and trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # === HTF: 1d data for choppiness index regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === 12h Indicators: Donchian channels (20-period) for trend direction ===
    lookback_12h = 20
    highest_high_12h = pd.Series(high_12h).rolling(window=lookback_12h, min_periods=lookback_12h).max().values
    lowest_low_12h = pd.Series(low_12h).rolling(window=lookback_12h, min_periods=lookback_12h).min().values
    
    # === 1d Indicators: Choppiness Index (14-period) for regime detection ===
    chop_period = 14
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    # Sum of TR over period
    tr_sum = pd.Series(tr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    # Highest high and lowest low over period
    hh_1d = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    ll_1d = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    # Choppiness Index: CHOP = 100 * log10(sumTR/(period*(HH-LL))) / log10(period)
    # Avoid division by zero
    hl_range = hh_1d - ll_1d
    chop_raw = np.ones_like(close_1d) * 50.0  # default to neutral
    mask = (hl_range > 0) & (~np.isnan(tr_sum))
    chop_raw[mask] = 100 * np.log10(tr_sum[mask] / (chop_period * hl_range[mask])) / np.log10(chop_period)
    chop_raw = np.where(chop_raw > 100, 100, chop_raw)
    chop_raw = np.where(chop_raw < 0, 0, chop_raw)
    
    # Align HTF indicators to 4h timeframe
    highest_high_12h_aligned = align_htf_to_ltf(prices, df_12h, highest_high_12h)
    lowest_low_12h_aligned = align_htf_to_ltf(prices, df_12h, lowest_low_12h)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # === 4h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_4h = 20
    highest_high_4h = pd.Series(high).rolling(window=lookback_4h, min_periods=lookback_4h).max().values
    lowest_low_4h = pd.Series(low).rolling(window=lookback_4h, min_periods=lookback_4h).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
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
    
    warmup = max(50, lookback_12h, lookback_4h, chop_period + 1, 20, 14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high_4h[i]) or np.isnan(lowest_low_4h[i]) or
            np.isnan(highest_high_12h_aligned[i]) or np.isnan(lowest_low_12h_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below 4h Donchian low (structure break)
                elif price < lowest_low_4h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above 4h Donchian high (structure break)
                elif price > highest_high_4h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        # Regime filter: only trade when market is trending (CHOP < 40)
        trending_regime = chop_aligned[i] < 40.0
        
        if volume_spike and trending_regime:
            # Determine 12h trend bias
            price_vs_12h_mid = price - (highest_high_12h_aligned[i] + lowest_low_12h_aligned[i]) / 2.0
            
            # Long entry: price breaks above 4h Donchian high with bullish 12h bias
            if (price > highest_high_4h[i] and 
                price_vs_12h_mid > 0):  # Above 12h midpoint = bullish bias
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 4h Donchian low with bearish 12h bias
            elif (price < lowest_low_4h[i] and 
                  price_vs_12h_mid < 0):  # Below 12h midpoint = bearish bias
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