#!/usr/bin/env python3
"""
Experiment #246: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts capture strong momentum moves, while HMA(21) filters for trend alignment and volume confirmation ensures institutional participation. This combination works in both bull and bear markets by trading breakouts in the direction of the medium-term trend. ATR-based stoploss manages risk. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_1d_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d data for trend strength regime
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr_1d = np.zeros(len(close_1d))
        tr_1d[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
        
        # Directional Movement
        dm_plus_1d = np.zeros(len(close_1d))
        dm_minus_1d = np.zeros(len(close_1d))
        for i in range(1, len(close_1d)):
            dm_plus_1d[i] = max(high_1d[i] - high_1d[i-1], 0)
            dm_minus_1d[i] = max(low_1d[i-1] - low_1d[i], 0)
        
        # Wilder's smoothing
        def wilders_smooth(series, period):
            result = np.full_like(series, np.nan)
            if len(series) < period:
                return result
            result[period-1] = np.nansum(series[:period])
            for i in range(period, len(series)):
                result[i] = result[i-1] - (result[i-1] / period) + series[i]
            return result
        
        atr_1d = wilders_smooth(tr_1d, 14)
        dm_plus_smooth = wilders_smooth(dm_plus_1d, 14)
        dm_minus_smooth = wilders_smooth(dm_minus_1d, 14)
        
        # DI+ and DI-
        di_plus_1d = np.zeros(len(close_1d))
        di_minus_1d = np.zeros(len(close_1d))
        valid = atr_1d > 0
        di_plus_1d[valid] = 100 * dm_plus_smooth[valid] / atr_1d[valid]
        di_minus_1d[valid] = 100 * dm_minus_smooth[valid] / atr_1d[valid]
        
        # DX and ADX
        dx_1d = np.zeros(len(close_1d))
        dx_valid = (di_plus_1d + di_minus_1d) > 0
        dx_1d[dx_valid] = 100 * np.abs(di_plus_1d[dx_valid] - di_minus_1d[dx_valid]) / (di_plus_1d[dx_valid] + di_minus_1d[dx_valid])
        
        adx_1d = np.full(len(close_1d), np.nan)
        for i in range(27, len(dx_1d)):  # ADX needs 2*period-1 values for stability
            if not np.isnan(dx_1d[i]):
                if i == 27:
                    adx_1d[i] = np.nanmean(dx_1d[14:i+1])
                else:
                    adx_1d[i] = (adx_1d[i-1] * 13 + dx_1d[i]) / 14
        
        # Align to 4h timeframe
        adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    else:
        adx_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Hull Moving Average(21)
    def wma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    def hma(data, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if len(data) < period:
            return np.full_like(data, np.nan)
        wma_half = wma(data, half_period)
        wma_full = wma(data, period)
        raw_hma = 2 * wma_half - wma_full
        hma_result = wma(raw_hma, sqrt_period)
        return hma_result
    
    hma_21 = hma(close, 21)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 25 (trending market) ---
        if adx_1d_aligned[i] < 25:
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Above average volume ---
        avg_volume = np.mean(volume[max(0, i-20):i]) if i >= 20 else volume[i]
        volume_confirm = volume[i] > avg_volume * 1.5
        
        # --- Donchian Breakout Signals ---
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # --- HMA Trend Filter ---
        price_above_hma = close[i] > hma_21[i]
        price_below_hma = close[i] < hma_21[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = np.zeros(i+1)
            atr_14[0] = tr[0]
            for j in range(1, i+1):
                atr_14[j] = (atr_14[j-1] * 13 + tr[j]) / 14
            current_atr = atr_14[i]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * current_atr
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * current_atr
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + price above HMA + volume confirmation
        if breakout_up and price_above_hma and volume_confirm:
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Donchian breakout down + price below HMA + volume confirmation
        elif breakout_down and price_below_hma and volume_confirm:
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>