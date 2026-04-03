#!/usr/bin/env python3
"""
Experiment #332: 12h Donchian Breakout + Volume + Chop Regime

HYPOTHESIS: 12h Donchian channel (20) breakouts with volume confirmation (>1.5x average) 
and chop regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) capture strong momentum 
while avoiding whipsaws. In ranging markets (CHOP > 61.8), fade at Donchian bands with 
volume confirmation. 12h timeframe targets 12-37 trades/year (50-150 total over 4 years) 
to minimize fee drag. Works in bull (breakouts with volume) and bear (failed reversals 
at bands) markets. Uses 1d/1w HTF for regime context.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_332_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for chop regime and trend context (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for chop regime proxy (ADX < 20 = ranging/chop, ADX > 25 = trending)
    def calculate_adx(high, low, close, period=14):
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
        
        atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
        return adx
    
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d EMA200 for long-term trend filter
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === HTF: 1w data for major trend context ===
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # === 12h Indicators: Donchian Channel (20) ===
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 12h Indicators: Choppiness Index (14) for regime detection ===
    def calculate_chop(high, low, close, period=14):
        """Calculate Choppiness Index: higher = more choppy/ranging, lower = more trending"""
        tr = np.zeros(len(high))
        for i in range(len(high)):
            if i == 0:
                tr[i] = high[i] - low[i]
            else:
                tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        chop = np.zeros(len(high))
        for i in range(len(high)):
            if atr_sum[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(period)
            else:
                chop[i] = 50.0  # Neutral when undefined
        return chop
    
    chop_12h = calculate_chop(high, low, close)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Warmup for indicators stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop_12h[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filters ---
        is_choppy = chop_12h[i] > 61.8  # Range/chop regime
        is_trending = chop_12h[i] < 38.2  # Strong trend regime
        is_neutral = chop_12h[i] >= 38.2 and chop_12h[i] <= 61.8  # Transition regime
        
        # --- 1d Trend Filter: Only trade in direction of 1d EMA200 ---
        trend_up = close[i] > ema200_1d_aligned[i]
        trend_down = close[i] < ema200_1d_aligned[i]
        
        # --- 1w Trend Filter: Only trade in direction of 1w EMA50 ---
        weekly_trend_up = close[i] > ema50_1w_aligned[i]
        weekly_trend_down = close[i] < ema50_1w_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Price Levels ---
        price = close[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        mid_band = donchian_mid[i]
        
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
            
            # Exit conditions based on regime
            if is_choppy:
                # In choppy markets, exit at mid-band
                if position_side > 0 and price < mid_band:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price > mid_band:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            elif is_trending:
                # In trending markets, trail with 2.5*ATR
                if position_side > 0 and price < entry_price - 2.0 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price > entry_price + 2.0 * atr_14[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 1 bar to reduce churn
            if bars_since_entry < 1:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price > upper band + volume spike + trend alignment
        long_breakout = (price > upper_band) and volume_spike and trend_up and weekly_trend_up
        
        # Short breakout: Price < lower band + volume spike + trend alignment
        short_breakout = (price < lower_band) and volume_spike and trend_down and weekly_trend_down
        
        # Long mean reversion in chop: Price < lower band + volume spike + chop
        long_reversion = (price < lower_band) and volume_spike and is_choppy
        
        # Short mean reversion in chop: Price > upper band + volume spike + chop
        short_reversion = (price > upper_band) and volume_spike and is_choppy
        
        if long_breakout or long_reversion:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_breakout or short_reversion:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals