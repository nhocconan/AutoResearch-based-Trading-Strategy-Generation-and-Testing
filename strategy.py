#!/usr/bin/env python3
"""
Experiment #309: 4h Donchian(20) Breakout + 1d Volume Spike + Choppiness Regime Filter

HYPOTHESIS: 4h Donchian breakouts filtered by 1d volume confirmation (>2.0x average) and 
choppiness regime (CHOP > 61.8 = ranging, CHOP < 38.2 = trending) capture strong momentum 
moves while avoiding whipsaws. In trending regimes (CHOP < 38.2), breakouts have higher 
follow-through. In ranging regimes (CHOP > 61.8), we fade breakouts at channel extremes. 
This adaptive approach works in both bull (trending breakouts) and bear (failed breaks reverse) 
markets. Uses ATR-based stoploss for risk management. Targets 19-50 trades/year (75-200 total).

IMPLEMENTATION NOTES:
- Uses discrete position sizing (0.25) to minimize churn
- Volume confirmation threshold set to 2.0x (balanced frequency)
- Choppiness regime calculated on 1d timeframe for stability
- Minimum holding period of 2 bars to reduce churn
- Warmup period set to 100 bars for stable indicators
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_309_4h_donchian_1d_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume MA and choppiness (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = np.zeros(len(df_1d))
    vol_ratio_1d[20:] = df_1d['volume'].values[20:] / vol_ma_20_1d[20:]
    vol_ratio_1d[:20] = 1.0  # Neutral for warmup
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 1d Indicators: Choppiness Index (CHOP) ===
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period:
            return np.full_like(high_arr, np.nan)
        tr = np.zeros(len(high_arr))
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(high_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        hh = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        chop = np.zeros(len(high_arr))
        for i in range(period-1, len(high_arr)):
            if hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50.0  # Neutral when range is zero
        return chop
    
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_h = np.full(n, np.nan)
    donchian_l = np.full(n, np.nan)
    donchian_m = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_h[i] = np.max(high[i-20:i])
        donchian_l[i] = np.min(low[i-20:i])
        donchian_m[i] = (donchian_h[i] + donchian_l[i]) / 2
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Increased warmup for stable HTF alignment and indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Choppiness Regime Filter ---
        chop_value = chop_1d_aligned[i]
        is_trending = chop_value < 38.2  # Trending regime
        is_ranging = chop_value > 61.8   # Ranging regime
        is_neutral = ~(is_trending | is_ranging)  # Neutral regime
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] < donchian_m[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian middle line reversion (take profit)
                if close[i] > donchian_m[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Adaptive logic based on regime:
        # In trending regime: trade breakouts in direction of momentum
        # In ranging regime: fade breakouts at extremes (mean reversion)
        # In neutral regime: standard breakout following
        
        if is_trending:
            # Trending: follow breakouts with volume confirmation
            long_condition = breakout_up and volume_spike
            short_condition = breakout_down and volume_spike
        elif is_ranging:
            # Ranging: fade extremes (donchian touches)
            touch_up = high[i] >= donchian_h[i]  # Touch or break upper band
            touch_down = low[i] <= donchian_l[i]  # Touch or break lower band
            # Fade: short at upper touch, long at lower touch
            long_condition = touch_down and volume_spike
            short_condition = touch_up and volume_spike
        else:
            # Neutral: standard breakout with volume confirmation
            long_condition = breakout_up and volume_spike
            short_condition = breakout_down and volume_spike
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals