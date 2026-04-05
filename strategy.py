#!/usr/bin/env python3
"""
exp_7111_6h_ichimoku_cloud_1d_regime_v1
Hypothesis: 6h Ichimoku cloud with 1d trend filter for regime adaptation.
In bull 1d regime (price > 1d Kumo): long on 6h TK cross above cloud, short on cross below.
In bear 1d regime (price < 1d Kumo): short on 6h TK cross below cloud, long on cross above.
Uses 6h volume confirmation to reduce false signals. Designed for 6h timeframe to capture
swings with ~12-37 trades/year (50-150 total over 4 years). Works in both bull and bear
markets by adapting to 1d Ichimoku cloud as dynamic support/resistance.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7111_6h_ichimoku_cloud_1d_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
# Ichimoku parameters (standard)
TENKAN_PERIOD = 9   # Conversion line
KIJUN_PERIOD = 26   # Base line
SENKOU_B_PERIOD = 52 # Leading span B
DISPLACEMENT = 26   # Kumo cloud displacement
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~20 * 6h = 5 days

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Ichimoku trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Ichimoku components for trend regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_1d = (pd.Series(high_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() +
                 pd.Series(low_1d).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # 1d Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_1d = (pd.Series(high_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() +
                pd.Series(low_1d).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # 1d Senkou Span A (Leading Span A): (Conversion Line + Base Line)/2
    senkou_span_a_1d = (tenkan_1d + kijun_1d) / 2
    # 1d Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    senkou_span_b_1d = (pd.Series(high_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() +
                        pd.Series(low_1d).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    # Kumo cloud boundaries (shifted forward by 26 periods)
    senkou_span_a_1d_shifted = np.roll(senkou_span_a_1d, DISPLACEMENT)
    senkou_span_b_1d_shifted = np.roll(senkou_span_b_1d, DISPLACEMENT)
    # Fill displaced values with NaN for alignment
    senkou_span_a_1d_shifted[:DISPLACEMENT] = np.nan
    senkou_span_b_1d_shifted[:DISPLACEMENT] = np.nan
    
    # Align 1d Ichimoku to 6h
    tenkan_1d_aligned = align_htf_to_ltf(prices, df_1d, tenkan_1d.values)
    kijun_1d_aligned = align_htf_to_ltf(prices, df_1d, kijun_1d.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_a_1d_shifted)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_span_b_1d_shifted)
    
    # Calculate 6h Ichimoku components for entry signals
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 6h Tenkan-sen (Conversion Line)
    tenkan_6h = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() +
                 pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    # 6h Kijun-sen (Base Line)
    kijun_6h = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() +
                pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    # TK Cross (Tenkan/Kijun crossover)
    tk_cross = tenkan_6h - kijun_6h
    tk_cross_prev = np.roll(tk_cross, 1)
    tk_cross_prev[0] = 0
    # Bullish TK cross: Tenkan crosses above Kijun
    tk_cross_up = (tk_cross > 0) & (tk_cross_prev <= 0)
    # Bearish TK cross: Tenkan crosses below Kijun
    tk_cross_down = (tk_cross < 0) & (tk_cross_prev >= 0)
    
    # 6h Kumo cloud (using same parameters as 1d but on 6h data)
    senkou_span_a_6h = (tenkan_6h + kijun_6h) / 2
    senkou_span_b_6h = (pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() +
                        pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2
    senkou_span_a_6h_shifted = np.roll(senkou_span_a_6h, DISPLACEMENT)
    senkou_span_b_6h_shifted = np.roll(senkou_span_b_6h, DISPLACEMENT)
    senkou_span_a_6h_shifted[:DISPLACEMENT] = np.nan
    senkou_span_b_6h_shifted[:DISPLACEMENT] = np.nan
    
    # Cloud top and bottom (for 6h)
    cloud_top_6h = np.maximum(senkou_span_a_6h_shifted, senkou_span_b_6h_shifted)
    cloud_bottom_6h = np.minimum(senkou_span_a_6h_shifted, senkou_span_b_6h_shifted)
    # Price above/below 6h cloud
    price_above_cloud = close > cloud_top_6h
    price_below_cloud = close < cloud_bottom_6h
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, DISPLACEMENT,
                VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if (np.isnan(tenkan_1d_aligned[i]) or np.isnan(kijun_1d_aligned[i]) or
            np.isnan(senkou_span_a_aligned[i]) or np.isnan(senkou_span_b_aligned[i])):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine 1d Ichimoku regime (trend filter)
        # Bull 1d regime: price above 1d Kumo
        bull_1d_regime = close[i] > max(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        # Bear 1d regime: price below 1d Kumo
        bear_1d_regime = close[i] < min(senkou_span_a_aligned[i], senkou_span_b_aligned[i])
        
        # 6h Ichimoku signals with 1d regime filter
        # In bull 1d regime: look for longs on TK cross above cloud
        long_signal = (bull_1d_regime and tk_cross_up[i] and price_above_cloud[i] and vol_confirmed)
        # In bull 1d regime: look for shorts on TK cross below cloud (counter-trend, weaker)
        short_signal_bull = (bull_1d_regime and tk_cross_down[i] and price_below_cloud[i] and vol_confirmed)
        
        # In bear 1d regime: look for shorts on TK cross below cloud
        short_signal = (bear_1d_regime and tk_cross_down[i] and price_below_cloud[i] and vol_confirmed)
        # In bear 1d regime: look for longs on TK cross above cloud (counter-trend, weaker)
        long_signal_bear = (bear_1d_regime and tk_cross_up[i] and price_above_cloud[i] and vol_confirmed)
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal or long_signal_bear:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal or short_signal_bull:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals