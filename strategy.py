#!/usr/bin/env python3
"""
Experiment #234: 1h HTF Direction + Volume Timing Strategy

HYPOTHESIS: Use 4h and 1d timeframes for signal direction (trend and regime) and 1h only for entry timing with volume confirmation. 4h Donchian(20) breakout provides trend direction, 1d ADX < 25 defines ranging regime for mean reversion at 4h VWAP, and 1h volume spike (>2.0x) triggers entry. This reduces trades by requiring confluence across three timeframes while capturing both trending and ranging market moves. Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_234_1h_htf_direction_volume_timing_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for trend direction (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Donchian(20) for trend direction
    def calculate_donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donch_4h_high, donch_4h_low = calculate_donchian_channels(
        df_4h['high'].values, df_4h['low'].values
    )
    donch_4h_high_aligned = align_htf_to_ltf(prices, df_4h, donch_4h_high)
    donch_4h_low_aligned = align_htf_to_ltf(prices, df_4h, donch_4h_low)
    
    # === HTF: 1d data for regime detection (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for regime detection (trending vs ranging)
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX (Average Directional Index)"""
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d VWAP for mean reversion reference
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3.0
    vwap_1d = (pd.Series(typical_price_1d * df_1d['volume'].values).cumsum() / 
               pd.Series(df_1d['volume'].values).cumsum()).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === 1h Indicators: ATR(14) for stoploss and VWAP calculation ===
    tr_1h = np.zeros(n)
    tr_1h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_1h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate 1h VWAP for entry timing
    typical_price_1h = (high + low + close) / 3.0
    vwap_1h = (pd.Series(typical_price_1h * volume).cumsum() / 
               pd.Series(volume).cumsum()).values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Session filter: 08-20 UTC ===
    hours = prices.index.hour  # Already datetime64, .hour works
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Position sizing (20% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for indicators stability
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donch_4h_high_aligned[i]) or np.isnan(donch_4h_low_aligned[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(vwap_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vwap_1h[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- HTF Direction: 4h Donchian breakout ====
        price_4h_high = donch_4h_high_aligned[i]
        price_4h_low = donch_4h_low_aligned[i]
        
        # --- HTF Regime: 1d ADX < 25 = ranging, > 25 = trending ---
        is_ranging = adx_1d_aligned[i] < 25
        is_trending = adx_1d_aligned[i] >= 25
        
        # --- Price ---
        price = close[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
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
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Exit conditions based on regime
            if is_ranging:
                # In ranging: exit when price returns to 1d VWAP
                if position_side > 0 and price >= vwap_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price <= vwap_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:
                # In trending: exit on Donchian reversal
                if position_side > 0 and price < price_4h_low:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price > price_4h_high:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: Require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if not volume_spike:
            signals[i] = 0.0
            continue
        
        # Ranging market logic: Mean reversion at 4h levels toward 1d VWAP
        if is_ranging:
            # Long: Price near 4h low AND below 1d VWAP (oversold)
            long_range = (price <= price_4h_low * 1.005) and (price < vwap_1d_aligned[i])
            
            # Short: Price near 4h high AND above 1d VWAP (overbought)
            short_range = (price >= price_4h_high * 0.995) and (price > vwap_1d_aligned[i])
            
            if long_range:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_range:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        
        # Trending market logic: Donchian breakout with 1h VWAP pullback
        else:  # is_trending
            # Long: Price breaks above 4h Donchian high AND pulls back to 1h VWAP
            long_trend = (price > price_4h_high) and (price <= vwap_1h[i] * 1.01)
            
            # Short: Price breaks below 4h Donchian low AND pulls back to 1h VWAP
            short_trend = (price < price_4h_low) and (price >= vwap_1h[i] * 0.99)
            
            if long_trend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_trend:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
    
    return signals