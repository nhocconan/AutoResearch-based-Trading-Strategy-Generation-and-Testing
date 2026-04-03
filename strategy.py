#!/usr/bin/env python3
"""
Experiment #213: 4h Donchian Breakout + 12h HMA Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 4h with volume confirmation and 12h HMA trend filter capture institutional breakout moves. Works in bull/bear by only trading in direction of higher timeframe trend (12h HMA). Volume spike (>2.0x average) confirms institutional participation. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_213_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA of half period
        weights_half = np.arange(1, half_period + 1)
        wma_half = pd.Series(arr).rolling(window=half_period, min_periods=half_period).apply(
            lambda x: np.dot(x, weights_half) / weights_half.sum(), raw=True
        ).values
        
        # WMA of full period
        weights_full = np.arange(1, period + 1)
        wma_full = pd.Series(arr).rolling(window=period, min_periods=period).apply(
            lambda x: np.dot(x, weights_full) / weights_full.sum(), raw=True
        ).values
        
        # HMA = WMA(2*WMA(half) - WMA(full)) with sqrt period
        raw_hma = 2 * wma_half - wma_full
        weights_sqrt = np.arange(1, sqrt_period + 1)
        hma = pd.Series(raw_hma).rolling(window=sqrt_period, min_periods=sqrt_period).apply(
            lambda x: np.dot(x, weights_sqrt) / weights_sqrt.sum(), raw=True
        ).values
        
        return hma
    
    close_12h = df_12h['close'].values
    hma_21 = calculate_hma(close_12h, 21)
    hma_trend_up = close_12h > hma_21
    hma_trend_down = close_12h < hma_21
    
    # Align to 4h timeframe
    hma_trend_up_aligned = align_htf_to_ltf(prices, df_12h, hma_trend_up)
    hma_trend_down_aligned = align_htf_to_ltf(prices, df_12h, hma_trend_down)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Donchian Channels (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_up_aligned[i]) or np.isnan(hma_trend_down_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches opposite Donchian band (mean reversion)
                if price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches opposite Donchian band (mean reversion)
                if price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume in uptrend
        if (price > donchian_high[i] and 
            hma_trend_up_aligned[i] and 
            volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian low with volume in downtrend
        elif (price < donchian_low[i] and 
              hma_trend_down_aligned[i] and 
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals