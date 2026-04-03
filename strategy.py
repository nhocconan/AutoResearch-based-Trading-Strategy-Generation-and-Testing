#!/usr/bin/env python3
"""
Experiment #068: 12h Donchian(20) Breakout + Weekly Trend Filter + Volume Spike

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by weekly trend direction 
(price above/below weekly HMA(21) = bullish/bearish bias), volume spikes (>2.0x average), 
and ATR-based stoploss capture strong momentum moves with reduced false breakouts. 
Weekly trend filter ensures trades align with higher-timeframe momentum, reducing whipsaws. 
12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag 
while capturing significant moves. Works in both bull (breakouts with volume) and bear 
(failed breaks reverse sharply) by using weekly trend as regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_068_12h_donchian_weekly_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA(21) for trend direction
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        if half_window < 1 or sqrt_window < 1:
            return np.full_like(values, np.nan)
        wma_half = np.full_like(values, np.nan)
        wma_full = np.full_like(values, np.nan)
        for i in range(len(values)):
            if i >= half_window - 1:
                wma_half[i] = wma(values[i - half_window + 1:i + 1], half_window)
            if i >= window - 1:
                wma_full[i] = wma(values[i - window + 1:i + 1], window)
        wma_diff = 2 * wma_half - wma_full
        hma_values = np.full_like(values, np.nan)
        for i in range(len(values)):
            if i >= window - 1 and i >= sqrt_window - 1 + window - 1:
                start_idx = i - sqrt_window + 1
                if start_idx >= 0 and start_idx < len(wma_diff):
                    hma_values[i] = wma(wma_diff[start_idx:i + 1], sqrt_window)
        return hma_values
    
    weekly_hma = hma(df_1w['close'].values, 21)
    weekly_hma_aligned = align_htf_to_ltf(prices, df_1w, weekly_hma)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 50  # Ensure enough data for HTF weekly HMA, ATR, and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(weekly_hma_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Trend Filter: Price above weekly HMA = bullish bias, below = bearish bias ---
        price_above_weekly_hma = close[i] > weekly_hma_aligned[i]
        price_below_weekly_hma = close[i] < weekly_hma_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + volume spike + price above weekly HMA
        long_condition = breakout_up and volume_spike and price_above_weekly_hma
        
        # Short: Donchian breakout down + volume spike + price below weekly HMA
        short_condition = breakout_down and volume_spike and price_below_weekly_hma
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals