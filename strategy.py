#!/usr/bin/env python3
"""
Experiment #4395: 6h Ichimoku Cloud + Weekly Trend + Volume Confirmation
HYPOTHESIS: Ichimoku TK cross (Tenkan/Kijun) aligned with weekly cloud color (bullish/bearish) and volume (>1.8x average) captures high-probability momentum shifts. Weekly cloud provides structural regime filter, reducing whipsaws in both bull and bear markets. Tenkan-Kijun cross acts as fast momentum signal, with cloud as dynamic support/resistance. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4395_6h_ichimoku1w_vol_v1"
timeframe = "6h"
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
    
    # === Precompute HTF: 1w Ichimoku Cloud ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 52:  # Need at least 52 weeks for proper calculation
        # Ichimoku components: Tenkan-sen (9-period), Kijun-sen (26-period), Senkou Span A/B (52-period)
        high_series = pd.Series(df_1w['high'].values)
        low_series = pd.Series(df_1w['low'].values)
        close_series = pd.Series(df_1w['close'].values)
        
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
        tenkan = (high_series.rolling(window=9, min_periods=9).max() + 
                  low_series.rolling(window=9, min_periods=9).min()) / 2
        # Kijun-sen (Base Line): (26-period high + 26-period low)/2
        kijun = (high_series.rolling(window=26, min_periods=26).max() + 
                 low_series.rolling(window=26, min_periods=26).min()) / 2
        # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
        senkou_b = ((high_series.rolling(window=52, min_periods=52).max() + 
                     low_series.rolling(window=52, min_periods=52).min()) / 2).shift(26)
        
        # Cloud color: bullish if Senkou A > Senkou B, bearish if Senkou A < Senkou B
        cloud_bullish = (senkou_a > senkou_b).values
        cloud_bearish = (senkou_a < senkou_b).values
        
        # Align to LTF (6h)
        tenkan_aligned = align_htf_to_ltf(prices, df_1w, tenkan.values)
        kijun_aligned = align_htf_to_ltf(prices, df_1w, kijun.values)
        cloud_bullish_aligned = align_htf_to_ltf(prices, df_1w, cloud_bullish.astype(float))
        cloud_bearish_aligned = align_htf_to_ltf(prices, df_1w, cloud_bearish.astype(float))
    else:
        tenkan_aligned = np.full(n, np.nan)
        kijun_aligned = np.full(n, np.nan)
        cloud_bullish_aligned = np.full(n, np.nan)
        cloud_bearish_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Ichimoku TK Cross ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_6h = (high_series.rolling(window=9, min_periods=9).max() + 
                 low_series.rolling(window=9, min_periods=9).min()) / 2
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_6h = (high_series.rolling(window=26, min_periods=26).max() + 
                low_series.rolling(window=26, min_periods=26).min()) / 2
    
    tenkan_6h_vals = tenkan_6h.values
    kijun_6h_vals = kijun_6h.values
    
    # TK Cross signals: bullish when Tenkan > Kijun, bearish when Tenkan < Kijun
    tk_bullish = tenkan_6h_vals > kijun_6h_vals
    tk_bearish = tenkan_6h_vals < kijun_6h_vals
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(26, 20, 14)  # TK cross, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(tenkan_6h_vals[i]) or np.isnan(kijun_6h_vals[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or
            np.isnan(cloud_bullish_aligned[i]) or np.isnan(cloud_bearish_aligned[i])):
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
        # Require volume confirmation (> 1.8x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.8
        
        # Weekly cloud regime: bullish if price above cloud, bearish if below cloud
        # For simplicity, use cloud color as regime filter
        weekly_bullish = cloud_bullish_aligned[i] > 0.5
        weekly_bearish = cloud_bearish_aligned[i] > 0.5
        
        # TK Cross conditions
        tk_cross_up = tk_bullish[i] and not tk_bullish[i-1]  # Tenkan crossed above Kijun
        tk_cross_down = tk_bearish[i] and not tk_bearish[i-1]  # Tenkan crossed below Kijun
        
        # Long conditions: bullish TK cross + bullish weekly cloud + volume
        long_entry = tk_cross_up and weekly_bullish and volume_confirm
        
        # Short conditions: bearish TK cross + bearish weekly cloud + volume
        short_entry = tk_cross_down and weekly_bearish and volume_confirm
        
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