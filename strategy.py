#!/usr/bin/env python3
"""
Experiment #7879: 6-hour Ichimoku Cloud + TK Cross with 1-day trend filter.
Hypothesis: Tenkan/Kijun cross (TK) with price above/below Kumo cloud (from 1d) provides high-probability entries with trend alignment. Kumo acts as dynamic support/resistance, reducing whipsaw in ranging markets. Trend filter ensures trades align with higher timeframe direction. Targets 50-150 trades over 4 years with controlled risk via ATR stops.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7879_6h_ichimoku_tk_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
KUMO_SHIFT = 26
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0
SIGNAL_SIZE = 0.25

def calculate_ichimoku(high, low):
    """Calculate Ichimoku components"""
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    senkou_a = ((tenkan + kijun) / 2)
    senkou_b = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() + 
                pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    return tenkan, kijun, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Ichimoku on daily
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(high_1d, low_1d)
    
    # Kumo cloud boundaries (shifted forward)
    senkou_a_shifted = np.roll(senkou_a_1d, KUMO_SHIFT)
    senkou_b_shifted = np.roll(senkou_b_1d, KUMO_SHIFT)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_shifted, senkou_b_shifted)
    cloud_bottom = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # Price relative to cloud: above = bullish, below = bearish, inside = neutral
    price_vs_cloud = np.where(close_1d > cloud_top, 1, 
                             np.where(close_1d < cloud_bottom, -1, 0))
    
    # TK cross signal: 1 = bullish cross, -1 = bearish cross
    tk_cross = np.where(tenkan_1d > kijun_1d, 1, -1)
    tk_cross_prev = np.roll(tk_cross, 1)
    tk_cross_prev[0] = tk_cross_prev[1] if len(tk_cross_prev) > 1 else 0
    tk_bullish_cross = (tk_cross == 1) & (tk_cross_prev == -1)
    tk_bearish_cross = (tk_cross == -1) & (tk_cross_prev == 1)
    
    # Align to LTF
    price_vs_cloud_aligned = align_htf_to_ltf(prices, df_1d, price_vs_cloud)
    tk_bullish_aligned = align_htf_to_ltf(prices, df_1d, tk_bullish_cross.astype(float))
    tk_bearish_aligned = align_htf_to_ltf(prices, df_1d, tk_bearish_cross.astype(float))
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    target_price = 0.0
    
    # Start from warmup period
    start = max(SENKOU_B_PERIOD + KUMO_SHIFT, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_cloud_aligned[i]) or np.isnan(tk_bullish_aligned[i]) or np.isnan(tk_bearish_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss or target
        if position == 1:  # long position
            if close[i] <= stop_price or close[i] >= target_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price or close[i] <= target_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine signals
        bullish_signal = (tk_bullish_aligned[i] == 1) and (price_vs_cloud_aligned[i] == 1)
        bearish_signal = (tk_bearish_aligned[i] == 1) and (price_vs_cloud_aligned[i] == -1)
        
        # Generate signals
        if position == 0:
            if bullish_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif bearish_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price - (ATR_TARGET_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
</p>