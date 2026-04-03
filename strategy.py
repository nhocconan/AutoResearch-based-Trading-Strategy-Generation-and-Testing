#!/usr/bin/env python3
"""
Experiment #257: 4h Donchian(20) Breakout + 1d EMA Trend + Volume Spike + 1w Chop Regime Filter

HYPOTHESIS: 4h Donchian breakouts filtered by 1d EMA50 trend, volume spikes (>2.0x average), 
and 1w choppiness regime (CHOP > 61.8 = range, CHOP < 38.2 = trend) capture strong momentum 
with reduced false breakouts. In trending regimes (CHOP < 38.2), we follow Donchian breakouts 
with EMA filter. In ranging regimes (CHOP > 61.8), we fade Donchian touches at channels with 
volume confirmation. This adaptive approach works in both bull/bear markets. Targets 19-50 
trades/year via tight entry conditions. Uses ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_257_4h_donchian_1d_ema_1w_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # === HTF: 1w data for Choppy Index regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate Choppiness Index (CHOP) on 1w data
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        if len(high_arr) < period:
            return np.full_like(high_arr, 50.0)  # Neutral chop value
        tr1 = np.abs(high_arr[1:] - low_arr[:-1])
        tr2 = np.abs(high_arr[1:] - close_arr[:-1])
        tr3 = np.abs(low_arr[1:] - close_arr[:-1])
        tr = np.maximum.reduce([tr1, tr2, tr3])
        tr = np.concatenate([[np.nan], tr])  # Align with original array
        atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
        highest_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        chop = 100 * np.log10(atr * np.sqrt(period) / (highest_high - lowest_low)) / np.log10(period)
        return np.where((highest_high - lowest_low) == 0, 50.0, chop)
    
    chop_1w = calculate_chop(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
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
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Ensure enough data for HTF EMA, CHOP, ATR, and Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(ema_1d_50_aligned[i]) or np.isnan(chop_1w_aligned[i]) or 
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: 1w Choppy Index ---
        chop_value = chop_1w_aligned[i]
        is_trending = chop_value < 38.2   # Strong trend regime
        is_ranging = chop_value > 61.8    # Strong ranging regime
        is_transitional = ~(is_trending | is_ranging)  # Mixed regime
        
        # --- 1d EMA50 Trend Filter ---
        price_above_ema = close[i] > ema_1d_50_aligned[i]
        price_below_ema = close[i] < ema_1d_50_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_h[i]
        breakout_down = close[i] < donchian_l[i]
        
        # --- Donchian Touch Conditions (for ranging regime fade) ---
        touch_upper = np.abs(close[i] - donchian_h[i]) < (donchian_h[i] - donchian_l[i]) * 0.02
        touch_lower = np.abs(close[i] - donchian_l[i]) < (donchian_h[i] - donchian_l[i]) * 0.02
        
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
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Regime-based exit logic
            if is_trending:
                # In trending regime: exit on Donchian middle reversion
                if (position_side > 0 and close[i] < donchian_m[i]) or \
                   (position_side < 0 and close[i] > donchian_m[i]):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:
                # In ranging/transitional regime: exit on opposite channel touch
                if (position_side > 0 and touch_lower) or \
                   (position_side < 0 and touch_upper):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_trending:
            # Trending regime: follow Donchian breakouts with EMA filter
            long_condition = breakout_up and volume_spike and price_above_ema
            short_condition = breakout_down and volume_spike and price_below_ema
        elif is_ranging:
            # Ranging regime: fade Donchian touches with volume confirmation
            long_condition = touch_lower and volume_spike and price_above_ema
            short_condition = touch_upper and volume_spike and price_below_ema
        else:
            # Transitional regime: require both EMA and chop alignment
            long_condition = breakout_up and volume_spike and price_above_ema and chop_value < 50
            short_condition = breakout_down and volume_spike and price_below_ema and chop_value > 50
        
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