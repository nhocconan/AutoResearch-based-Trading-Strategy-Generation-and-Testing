#!/usr/bin/env python3
"""
4h_1d_inside_bar_breakout_v1
Strategy: 4h inside bar breakout with 1d trend filter
Timeframe: 4h
Leverage: 1.0
Hypothesis: Inside bars indicate consolidation. Breakout from inside bar range with volume confirmation and aligned with 1d EMA trend yields high-probability trades. Works in bull/bear by only trading in direction of higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_inside_bar_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4x ATR for volatility filter (not used but placeholder)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 4h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d EMA50 (trend filter) ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Inside bar detection on 4h ===
    # Inside bar: current high < previous high AND current low > previous low
    inside_bar = (high < np.roll(high, 1)) & (low > np.roll(low, 1))
    inside_bar[0] = False  # First bar has no previous
    
    # Inside bar range
    inside_high = np.where(inside_bar, high, np.nan)
    inside_low = np.where(inside_bar, low, np.nan)
    
    # Forward fill inside bar ranges until broken
    inside_high_ff = pd.Series(inside_high).ffill().values
    inside_low_ff = pd.Series(inside_low).ffill().values
    
    # Breakout conditions
    breakout_up = (high > inside_high_ff) & inside_bar
    breakout_down = (low < inside_low_ff) & inside_bar
    
    # Session filter: 0-23 UTC (covers major sessions)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 0) & (hours <= 23)
    
    # Minimum holding period: 2 bars (8 hours) to reduce churn
    hold_count = np.zeros(n, dtype=int)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Decrease hold counter
        if hold_count[i] > 0:
            hold_count[i] -= 1
        
        # Skip if any required data is invalid or outside session or holding
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(vol_ma_20[i]) or
            not in_session[i] or hold_count[i] > 0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_20[i]
        ema_50 = ema_50_1d_aligned[i]
        
        # Volume confirmation: 4h volume must be elevated
        volume_confirmed = volume_current > 1.5 * vol_ma
        
        # Trend filter: price must be above/below 1d EMA50
        uptrend_1d = price_close > ema_50
        downtrend_1d = price_close < ema_50
        
        # Long conditions: breakout up from inside bar with volume + 1d uptrend
        long_signal = volume_confirmed and breakout_up[i] and uptrend_1d
        
        # Short conditions: breakout down from inside bar with volume + 1d downtrend
        short_signal = volume_confirmed and breakout_down[i] and downtrend_1d
        
        # Exit when price returns to the inside bar midpoint (mean reversion)
        inside_mid = (inside_high_ff[i] + inside_low_ff[i]) / 2
        exit_long = position == 1 and price_close < inside_mid
        exit_short = position == -1 and price_close > inside_mid
        
        # Trading logic with minimum holding period
        if long_signal and position != 1:
            position = 1
            hold_count[i] = 2  # Hold for 2 bars minimum
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            hold_count[i] = 2  # Hold for 2 bars minimum
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Inside bars indicate consolidation. Breakout from inside bar range with volume confirmation and aligned with 1d EMA trend yields high-probability trades. Works in bull/bear by only trading in direction of higher timeframe trend.