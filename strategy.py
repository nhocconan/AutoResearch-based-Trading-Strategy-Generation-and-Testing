#!/usr/bin/env python3
# 12h_W1_Engulfing_Pullback
# Hypothesis: Combines weekly bullish/bearish engulfing patterns with 12h price pullback to weekly VWAP for high-probability entries.
# Weekly bullish engulfing + price > weekly VWAP + pullback (close < 12h EMA20) = long.
# Weekly bearish engulfing + price < weekly VWAP + pullback (close > 12h EMA20) = short.
# Uses weekly timeframe for structure (engulfing/VWAP) and 12h for timing (EMA pullback).
# Works in bull markets by buying dips in uptrends and in bear by selling rallies in downtrends.
# Low-frequency signal (~15-25 trades/year) minimizes fee drag.

name = "12h_W1_Engulfing_Pullback"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for engulfing and VWAP
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly bullish/bearish engulfing ---
    open_1w = df_1w['open'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    bullish_engulf = (close_1w > open_1w) & (open_1w < close_1w) & \
                     (close_1w > open_1w) & (open_1w < close_1w) & \
                     (close_1w >= open_1w) & (open_1w <= close_1w) & \
                     (close_1w > open_1w.shift(1)) & (open_1w < close_1w.shift(1))
    bearish_engulf = (close_1w < open_1w) & (open_1w > close_1w) & \
                     (close_1w < open_1w) & (open_1w > close_1w) & \
                     (close_1w <= open_1w) & (open_1w >= close_1w) & \
                     (close_1w < open_1w.shift(1)) & (open_1w > close_1w.shift(1))
    # Fix: proper engulfing conditions
    bullish_engulf = (close_1w > open_1w) & (open_1w < close_1w.shift(1)) & (close_1w > open_1w.shift(1)) & (open_1w < close_1w.shift(1))
    bearish_engulf = (close_1w < open_1w) & (open_1w > close_1w.shift(1)) & (close_1w < open_1w.shift(1)) & (open_1w > close_1w.shift(1))
    # First bar: no previous week
    bullish_engulf[0] = False
    bearish_engulf[0] = False
    
    # --- Weekly VWAP (volume-weighted average price) ---
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vol_price_1w = typical_price_1w * volume_1w if 'volume_1w' in df_1w.columns else typical_price_1w * df_1w['volume'].values
    cum_vol_price = np.cumsum(vol_price_1w)
    cum_vol = np.cumsum(df_1w['volume'].values)
    vwap_1w = np.divide(cum_vol_price, cum_vol, out=np.zeros_like(cum_vol_price), where=cum_vol!=0)
    
    # --- 12h EMA20 for pullback timing ---
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to 12h timeframe
    bullish_engulf_aligned = align_htf_to_ltf(prices, df_1w, bullish_engulf)
    bearish_engulf_aligned = align_htf_to_ltf(prices, df_1w, bearish_engulf)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for weekly data and EMA20
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(bullish_engulf_aligned[i]) or
            np.isnan(bearish_engulf_aligned[i]) or
            np.isnan(vwap_1w_aligned[i]) or
            np.isnan(ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly conditions
        is_bullish_engulf = bullish_engulf_aligned[i]
        is_bearish_engulf = bearish_engulf_aligned[i]
        price_vs_vwap = close[i] - vwap_1w_aligned[i]
        
        # 12h pullback condition
        pullback_long = close[i] < ema_20[i]   # price below EMA20 = pullback in uptrend
        pullback_short = close[i] > ema_20[i]  # price above EMA20 = pullback in downtrend
        
        if position == 0:
            if is_bullish_engulf and price_vs_vwap > 0 and pullback_long:
                # Long: weekly bullish engulf + price above weekly VWAP + 12h pullback
                signals[i] = 0.25
                position = 1
            elif is_bearish_engulf and price_vs_vwap < 0 and pullback_short:
                # Short: weekly bearish engulf + price below weekly VWAP + 12h pullback
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: weekly bearish engulf OR price crosses below weekly VWAP
                if is_bearish_engulf or price_vs_vwap < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly bullish engulf OR price crosses above weekly VWAP
                if is_bullish_engulf or price_vs_vwap > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals