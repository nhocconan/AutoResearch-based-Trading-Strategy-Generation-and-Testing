#!/usr/bin/env python3
"""
Experiment #297: 4h Donchian(20) breakout + 1d HMA trend + volume confirmation
HYPOTHESIS: Price breaking 4h Donchian channels with 1d HMA trend filter and volume confirmation captures strong momentum moves. In bull markets, breakouts above upper channel with bullish 1d HMA trend continue up. In bear markets, breakouts below lower channel with bearish 1d HMA trend continue down. Volume confirmation reduces false breakouts. Discrete sizing (0.30) balances profit potential and drawdown control. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_297_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    close_1d = df_1d['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt_period = int(np.sqrt(period))
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        wma_2half = 2 * wma_half
        # Pad to same length
        raw_hma = wma_2half[-len(wma_full):] - wma_full
        hma_vals = wma(raw_hma, sqrt_period)
        # Return full array with NaN padding
        result = np.full_like(arr, np.nan)
        result[period-1:] = hma_vals[sqrt_period-1:]
        return result
    
    hma_21 = hma(close_1d, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # === 4h Indicators: Donchian(20) channels ===
    def donchian_channels(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_channel, lower_channel = donchian_channels(high, low, 20)
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > upper_channel[i]
        breakout_down = price < lower_channel[i]
        
        # --- 1d HMA Trend Filter ---
        hma_trend_up = hma_21_aligned[i] > hma_21_aligned[i-1] if i > 0 else False
        hma_trend_down = hma_21_aligned[i] < hma_21_aligned[i-1] if i > 0 else False
        
        # --- Exit Logic: ATR-based stoploss and channel re-entry ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (using channel width as proxy)
                channel_width = upper_channel[i] - lower_channel[i]
                stop_level = entry_price - 2.5 * channel_width * 0.15  # approximate ATR
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit: return to middle of channel or opposite breakout
                if price < (upper_channel[i] + lower_channel[i]) / 2:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                channel_width = upper_channel[i] - lower_channel[i]
                stop_level = entry_price + 2.5 * channel_width * 0.15
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit: return to middle of channel
                if price > (upper_channel[i] + lower_channel[i]) / 2:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND bullish 1d HMA trend
            if breakout_up and hma_trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish 1d HMA trend
            elif breakout_down and hma_trend_down:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals