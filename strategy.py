#!/usr/bin/env python3
"""
Experiment #131: 6h Elder Ray + 1d Regime Filter

HYPOTHESIS: Elder Ray (Bull Power/Bear Power) measures buying/selling pressure relative to EMA13.
Combined with 1d ADX regime filter (ADX>25 = trend, ADX<20 = range) to avoid whipsaws.
In trend regimes: trade Elder Ray extremes in trend direction. In range regimes: fade Elder Ray extremes.
Uses 6h timeframe for balance of signal quality and trade frequency. Target: 75-150 total trades over 4 years.
Works in bull/bear by adapting to regime: trend following in strong trends, mean reversion in ranges.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_elder_ray_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime filter (ADX) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ADX on 1d data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            elif plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            else:
                minus_dm[i] = 0
            
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === 6h Indicators ===
    # EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Smoothed Elder Ray (13-period)
    bull_power_smooth = pd.Series(bull_power).ewm(span=13, min_periods=13, adjust=False).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema_13[i]) or np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d Regime Filter ---
        adx_val = adx_1d_aligned[i]
        is_trending = adx_val > 25
        is_ranging = adx_val < 20
        
        # --- Elder Ray Signals ---
        bull_extreme = bull_power_smooth[i] > np.percentile(bull_power_smooth[max(0, i-100):i+1], 80)
        bear_extreme = bear_power_smooth[i] < np.percentile(bear_power_smooth[max(0, i-100):i+1], 20)
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit conditions based on regime
            if is_trending:
                # In trend: exit when Elder Ray reverses
                if position_side > 0:  # Long
                    if bull_power_smooth[i] < 0:  # Bull power turned negative
                        in_position = False
                        position_side = 0
                else:  # Short
                    if bear_power_smooth[i] > 0:  # Bear power turned positive
                        in_position = False
                        position_side = 0
            else:  # ranging
                # In range: exit when Elder Ray returns to neutral
                if position_side > 0:  # Long
                    if bull_power_smooth[i] < np.percentile(bull_power_smooth[max(0, i-20):i+1], 50):
                        in_position = False
                        position_side = 0
                else:  # Short
                    if bear_power_smooth[i] > np.percentile(bear_power_smooth[max(0, i-20):i+1], 50):
                        in_position = False
                        position_side = 0
            
            if not in_position:
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_trending:
            # Trend regime: follow Elder Ray extremes
            if bull_extreme and close[i] > ema_13[i]:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            elif bear_extreme and close[i] < ema_13[i]:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
        else:  # ranging
            # Range regime: fade Elder Ray extremes
            if bull_extreme and close[i] < ema_13[i]:
                # Overbought in range -> short
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            elif bear_extreme and close[i] > ema_13[i]:
                # Oversold in range -> long
                in_position = True
                position_side = 1
                signals[i] = SIZE
    
    return signals