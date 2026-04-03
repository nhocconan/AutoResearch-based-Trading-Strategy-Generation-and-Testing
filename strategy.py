#!/usr/bin/env python3
"""
Experiment #237: 4h Donchian(20) Breakout + Daily Trend + Volume Spike + Chop Filter

HYPOTHESIS: 4h Donchian channel breakouts filtered by 1d EMA trend (price > EMA50 = bullish bias, 
price < EMA50 = bearish bias) and volume spikes (>2.0x average) capture strong momentum moves. 
Added choppiness regime filter (CHOP < 61.8 = trending) to avoid false signals in ranging markets. 
Uses ATR-based stoploss for risk management. Targets 19-50 trades/year to minimize fee drag.
Works in both bull (breakouts with volume) and bear (failed breaks reverse sharply) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_237_4h_donchian_daily_trend_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA50 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 from daily close
    ema_1d_50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align to LTF (4h) timeframe with shift(1) for completed bars only
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
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
    vol_ratio[:20] = 1.0
    
    # === 4h Indicators: Choppiness Index (14) for regime filter ===
    def true_range(high, low, close_prev):
        return np.maximum(high - low, np.maximum(abs(high - close_prev), abs(low - close_prev)))
    
    tr_chop = np.zeros(n)
    tr_chop[0] = high[0] - low[0]
    for i in range(1, n):
        tr_chop[i] = true_range(high[i], low[i], close[i-1])
    
    atr_14_chop = pd.Series(tr_chop).ewm(span=14, min_periods=14, adjust=False).mean().values
    sum_tr_14 = pd.Series(tr_chop).rolling(window=14, min_periods=14).sum().values
    max_h_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_l_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = np.full(n, np.nan)
    for i in range(14, n):
        if max_h_14[i] > min_l_14[i]:
            chop[i] = 100 * np.log10(sum_tr_14[i] / (max_h_14[i] - min_l_14[i])) / np.log10(14)
        else:
            chop[i] = 50.0  # Neutral when no range
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # Ensure enough data for HTF EMA, ATR, Donchian, Chop
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_h[i]) or np.isnan(donchian_l[i]) or 
            np.isnan(ema_1d_50_aligned[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Price > daily EMA50 = bullish bias, Price < daily EMA50 = bearish bias ---
        price_above_ema = close[i] > ema_1d_50_aligned[i]
        price_below_ema = close[i] < ema_1d_50_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Regime Filter: Only trade in trending markets (CHOP < 61.8) ---
        trending_market = chop[i] < 61.8
        
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
        # Long: Donchian breakout up + volume spike + price above daily EMA + trending market
        long_condition = breakout_up and volume_spike and price_above_ema and trending_market
        
        # Short: Donchian breakout down + volume spike + price below daily EMA + trending market
        short_condition = breakout_down and volume_spike and price_below_ema and trending_market
        
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