#!/usr/bin/env python3
"""
Experiment #230: 1d Donchian Breakout + Weekly Trend + Volume Spike

HYPOTHESIS: Daily Donchian(20) breakouts with weekly trend filter (price > weekly EMA50) 
and volume confirmation (>2x average) capture strong momentum moves in both bull and bear markets.
In ranging markets (weekly ADX < 25), we fade breaks of Donchian bands with volume confirmation 
and price deviation from weekly VWAP (>1.5 ATR). Weekly timeframe ensures we trade with the 
higher-trend structure while daily provides timely entries. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_230_1d_donchian_weekly_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend and regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly ADX for regime detection
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
    
    adx_1w = calculate_adx(df_1w['high'].values, df_1w['low'].values, df_1w['close'].values)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate weekly VWAP for mean reversion timing
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vwap_numerator = np.zeros(len(df_1w))
    vwap_denominator = np.zeros(len(df_1w))
    vwap_1w = np.full(len(df_1w), np.nan)
    
    for i in range(len(df_1w)):
        start_idx = max(0, i - 19)
        vol_sum = df_1w['volume'].iloc[start_idx:i+1].sum()
        if vol_sum > 0:
            tp_sum = (typical_price_1w[start_idx:i+1] * df_1w['volume'].iloc[start_idx:i+1]).sum()
            vwap_1w[i] = tp_sum / vol_sum
        else:
            vwap_1w[i] = typical_price_1w[i]
    
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # === 1d Indicators: Donchian channels (20-period) ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # === 1d Indicators: ATR(14) for stoploss and normalization ===
    tr_1d = np.zeros(n)
    tr_1d[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1d[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 1d Indicators: VWAP deviation for mean reversion ===
    typical_price_1d = (high + low + close) / 3.0
    vwap_numerator_1d = np.zeros(n)
    vwap_denominator_1d = np.zeros(n)
    vwap_1d = np.full(n, np.nan)
    
    for i in range(n):
        start_idx = max(0, i - 19)
        vol_sum = volume[start_idx:i+1].sum()
        if vol_sum > 0:
            tp_sum = (typical_price_1d[start_idx:i+1] * volume[start_idx:i+1]).sum()
            vwap_1d[i] = tp_sum / vol_sum
        else:
            vwap_1d[i] = typical_price_1d[i]
    
    # VWAP deviation normalized by ATR
    vwap_dev_1d = np.zeros(n)
    vwap_dev_1d[14:] = (close[14:] - vwap_1d[14:]) / atr_14[14:]
    vwap_dev_1d[:14] = 0.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Warmup for stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(vwap_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(vwap_dev_1d[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Regime Filter ---
        is_trending = adx_1w_aligned[i] > 25
        is_ranging = adx_1w_aligned[i] < 25
        
        # --- Volume Confirmation ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Price Levels ---
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        weekly_ema50 = ema50_1w_aligned[i]
        weekly_vwap = vwap_1w_aligned[i]
        vwap_dev = vwap_dev_1d[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Exit conditions based on regime
            if is_trending:
                # In trending markets: exit on Donchian opposite break
                if position_side > 0 and price < lower:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and price > upper:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # ranging
                # In ranging markets: exit on mean reversion to weekly VWAP
                if abs(vwap_dev) < 0.5:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 days to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry ---
        # Trending market logic: Donchian breakouts with volume and weekly trend alignment
        if is_trending:
            # Long breakout: price > upper Donchian + volume spike + price > weekly EMA50
            long_breakout = (price > upper) and volume_spike and (price > weekly_ema50)
            
            # Short breakout: price < lower Donchian + volume spike + price < weekly EMA50
            short_breakout = (price < lower) and volume_spike and (price < weekly_ema50)
            
            if long_breakout:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_breakout:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
        
        # Ranging market logic: fade Donchian breaks with volume and VWAP deviation
        else:
            # Long mean reversion: price < lower Donchian + volume spike + price below weekly VWAP
            long_mr = (price < lower) and volume_spike and (vwap_dev < -1.5)
            
            # Short mean reversion: price > upper Donchian + volume spike + price above weekly VWAP
            short_mr = (price > upper) and volume_spike and (vwap_dev > 1.5)
            
            if long_mr:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_mr:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
    
    return signals