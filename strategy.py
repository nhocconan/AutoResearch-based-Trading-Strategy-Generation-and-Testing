#!/usr/bin/env python3
"""
Experiment #985: 12h Donchian(20) Breakout + Volume Spike + Choppiness Filter
HYPOTHESIS: 12h Donchian breakouts capture institutional order flow with lower noise. 
Long when price breaks above upper band with volume spike (>1.8x avg) and choppy market (CHOP > 61.8). 
Short when price breaks below lower band with volume spike and choppy market. 
Uses 1d timeframe for trend alignment only (not entry). Discrete position sizing (0.25) minimizes fee churn.
Target: 75-150 total trades over 4 years (19-37/year) on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_985_12h_donchian20_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === 12h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Choppiness Index (CHOP) ===
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index"""
        if len(close_arr) < period:
            return np.full_like(close_arr, 50.0)
        atr_sum = np.zeros(n)
        for i in range(1, n):
            tr = max(high_arr[i] - low_arr[i], abs(high_arr[i] - close_arr[i-1]), abs(low_arr[i] - close_arr[i-1]))
            atr_sum[i] = atr_sum[i-1] + tr
        atr_sum[0] = high_arr[0] - low_arr[0]
        
        max_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        
        chop = np.full(n, 50.0)
        for i in range(period-1, n):
            if atr_sum[i] > 0 and (max_high[i] - min_low[i]) > 0:
                log_sum = np.log10(atr_sum[i] / (max_high[i] - min_low[i]))
                log_period = np.log10(period)
                chop[i] = 100 * log_sum / log_period
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for SMA and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(sma_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~3d on 12h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        # Choppiness filter: require choppy market (CHOP > 61.8) for mean reversion
        choppy_market = chop[i] > 61.8
        
        if volume_spike and choppy_market:
            # 1d trend filter: price vs 50 SMA
            price_above_sma = price > sma_1d_aligned[i]
            price_below_sma = price < sma_1d_aligned[i]
            
            # Breakout logic: price breaks Donchian bands
            if price > donch_high[i] and price_above_sma:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and price_below_sma:
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