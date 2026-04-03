#!/usr/bin/env python3
"""
Experiment #1914: 1h Donchian Breakout + 4h/1d Trend + Volume Confirmation + Session Filter
HYPOTHESIS: Donchian channel breakouts on 1h capture momentum, filtered by 4h/1d trend alignment and volume spikes. Session filter (08-20 UTC) reduces noise. Target: 60-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1914_1h_donchian20_4h_1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for Donchian and trend ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian(20)
    donch_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h EMA(50) for trend
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_4h = np.where(close_4h > ema_50_4h, 1, -1)
    
    # Align 4h indicators to 1h
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for higher timeframe trend ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(100) for trend filter
    ema_100_1d = pd.Series(close_1d).ewm(span=100, min_periods=100, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_100_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Session filter: 08-20 UTC ===
    # open_time is datetime64[ms], use index.hour
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # sufficient for EMA(100) and Donchian(20)
    
    for i in range(warmup, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # --- Data Validity Check ---
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below 4h Donchian low
                if price < donch_low_4h_aligned[i]:
                    exit_signal = True
                # Exit if 4h trend turns bearish
                elif trend_4h_aligned[i] < 0:
                    exit_signal = True
                # Exit if 1d trend turns bearish (stronger filter)
                elif trend_1d_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above 4h Donchian high
                if price > donch_high_4h_aligned[i]:
                    exit_signal = True
                # Exit if 4h trend turns bullish
                elif trend_4h_aligned[i] > 0:
                    exit_signal = True
                # Exit if 1d trend turns bullish
                elif trend_1d_aligned[i] > 0:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require both 4h and 1d trend alignment for bias filter
        trend_bias_4h = trend_4h_aligned[i]
        trend_bias_1d = trend_1d_aligned[i]
        
        # Require both trends to agree
        if trend_bias_4h == trend_bias_1d:
            # Volume confirmation: require volume spike (> 1.5x average)
            volume_spike = vol_ratio[i] > 1.5
            
            if volume_spike:
                # Long entry: price breaks above 4h Donchian high AND trends up
                if trend_bias_4h > 0 and price > donch_high_4h_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short entry: price breaks below 4h Donchian low AND trends down
                elif trend_bias_4h < 0 and price < donch_low_4h_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals