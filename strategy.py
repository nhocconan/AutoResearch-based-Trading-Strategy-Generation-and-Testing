#!/usr/bin/env python3
"""
Experiment #1912: 12h Donchian(20) Breakout + 1d Volume Spike + Chop Regime Filter
HYPOTHESIS: On 12h timeframe, Donchian channel breakouts with volume confirmation and 
choppiness regime filter capture institutional moves while avoiding whipsaws. 
Volume spike (>2x 20-period average) confirms breakout validity. 
Choppiness index (CHOP) > 61.8 triggers mean reversion at channel midpoints; 
CHOP < 38.2 enables trend-following breakouts. 
Uses discrete position sizing (0.25) to minimize fee churn. 
Target: 75-150 total trades over 4 years on BTC/ETH/SOL.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1912_12h_donchian20_1d_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume MA and chop regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # 1d volume MA(20) for spike detection
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.ones(len(close_1d))
    vol_ratio_1d[20:] = volume_1d[20:] / vol_ma_1d[20:]
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 1d Choppiness Index (CHOP) - measures trend vs range
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index: higher = more range-bound, lower = more trending"""
        atr_sum = np.zeros(len(close_arr))
        true_range = np.zeros(len(close_arr))
        
        # True Range
        true_range[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = abs(high_arr[i] - close_arr[i-1])
            lc = abs(low_arr[i] - close_arr[i-1])
            true_range[i] = max(hl, hc, lc)
        
        # ATR calculation using Wilder's smoothing
        atr = np.zeros(len(close_arr))
        atr[period-1] = np.mean(true_range[:period])
        for i in range(period, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros(len(close_arr))
        for i in range(period-1, len(close_arr)):
            if i >= period-1:
                atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros(len(close_arr))
        lowest_low = np.zeros(len(close_arr))
        for i in range(len(close_arr)):
            if i < period-1:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high_arr[i-period+1:i+1])
                lowest_low[i] = np.min(low_arr[i-period+1:i+1])
        
        # Chop calculation
        chop = np.full(len(close_arr), np.nan)
        for i in range(period-1, len(close_arr)):
            if atr_sum[i] > 0 and highest_high[i] > lowest_low[i]:
                log_term = np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i]))
                chop[i] = 100 * log_term / np.log10(period)
        
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, period=14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(donchian_period, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            chop_val = chop_1d_aligned[i]
            
            if position_side > 0:  # Long position
                # In trending regime (CHOP < 38.2): exit on Donchian midpoint break
                if chop_val < 38.2:
                    if price < donchian_mid[i]:
                        exit_signal = True
                # In ranging regime (CHOP > 61.8): exit at opposite Donchian band
                elif chop_val > 61.8:
                    if price < lowest_low[i]:
                        exit_signal = True
                # Transition regime: exit at midpoint
                else:
                    if price < donchian_mid[i]:
                        exit_signal = True
            else:  # Short position
                # In trending regime (CHOP < 38.2): exit on Donchian midpoint break
                if chop_val < 38.2:
                    if price > donchian_mid[i]:
                        exit_signal = True
                # In ranging regime (CHOP > 61.8): exit at opposite Donchian band
                elif chop_val > 61.8:
                    if price > highest_high[i]:
                        exit_signal = True
                # Transition regime: exit at midpoint
                else:
                    if price > donchian_mid[i]:
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
        # Volume confirmation: require volume spike (> 2x average)
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        chop_val = chop_1d_aligned[i]
        
        if volume_spike:
            # Determine market regime from Chop
            if chop_val < 38.2:
                # Trending regime: follow breakout direction
                # Long: price breaks above upper Donchian band
                if price > highest_high[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price breaks below lower Donchian band
                elif price < lowest_low[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            elif chop_val > 61.8:
                # Ranging regime: mean reversion at extremes
                # Long: price touches lower band and reverses up
                if price <= lowest_low[i] and i > warmup and close[i-1] > lowest_low[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short: price touches upper band and reverses down
                elif price >= highest_high[i] and i > warmup and close[i-1] < highest_high[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            # Transition regime (38.2 <= CHOP <= 61.8): no new entries to avoid whipsaws
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals