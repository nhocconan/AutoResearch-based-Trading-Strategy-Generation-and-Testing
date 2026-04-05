#!/usr/bin/env python3
"""
exp_7215_6h_ichimoku_kijun_regime_v1
Hypothesis: 6h Ichimoku Kijun-Sen (base line) with 1d/1w trend regime filter for ETH/BTC/SOL.
In strong uptrend (price > 1w Kumo top): long on pullback to Kijun-Sen with volume confirmation.
In strong downtrend (price < 1w Kumo bottom): short on rally to Kijun-Sen with volume confirmation.
In ranging (price inside 1w Kumo): fade at Kumo edges with volume confirmation.
Uses weekly Ichimoku for regime and 6h Kijun-Sen for dynamic support/resistance.
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to weekly Ichimoku-defined trend regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7215_6h_ichimoku_kijun_regime_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
KIJUN_PERIOD = 26          # Base line period
TENKAN_PERIOD = 9          # Conversion line period
SENKOU_SPAN_B_PERIOD = 52  # Leading span B period
KUMO_SHIFT = 26            # Kumo cloud forward shift
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 8          # ~2 days (8 * 6h = 48h)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for Ichimoku regime
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Ichimoku components
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
    tenkan_sen = (pd.Series(high_1w).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low_1w).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
    kijun_sen = (pd.Series(high_1w).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low_1w).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(KUMO_SHIFT)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high_1w).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).max() + 
                      pd.Series(low_1w).rolling(window=SENKOU_SPAN_B_PERIOD, min_periods=SENKOU_SPAN_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Align 1w Ichimoku to 6h
    tenkan_sen_aligned = align_htf_to_ltf(prices, df_1w, tenkan_sen.values)
    kijun_sen_aligned = align_htf_to_ltf(prices, df_1w, kijun_sen.values)
    senkou_span_a_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_a.values)
    senkou_span_b_aligned = align_htf_to_ltf(prices, df_1w, senkou_span_b.values)
    
    # Kumo (cloud) boundaries: Senkou Span A and B
    kumo_top = np.maximum(senkou_span_a_aligned, senkou_span_b_aligned)
    kumo_bottom = np.minimum(senkou_span_a_aligned, senkou_span_b_aligned)
    
    # Calculate LTF indicators (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Kijun-sen (base line) for dynamic support/resistance
    kijun_sen_6h = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                    pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
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
    start = max(KIJUN_PERIOD, TENKAN_PERIOD, SENKOU_SPAN_B_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + KUMO_SHIFT
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(kumo_top[i]) or np.isnan(kumo_bottom[i]):
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
        vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().iloc[i]
        vol_confirmed = not np.isnan(vol_ma) and volume[i] > vol_ma * VOL_BASE_THRESHOLD
        
        # Determine market regime based on weekly Kumo
        price_above_kumo = close[i] > kumo_top[i]
        price_below_kumo = close[i] < kumo_bottom[i]
        price_in_kumo = (close[i] >= kumo_bottom[i]) & (close[i] <= kumo_top[i])
        
        # 6h price relative to 6h Kijun-sen (dynamic support/resistance)
        price_above_kijun = close[i] > kijun_sen_6h.iloc[i]
        price_below_kijun = close[i] < kijun_sen_6h.iloc[i]
        
        # Entry logic
        if position == 0:  # flat - look for new entries
            # Strong uptrend: price above weekly Kumo -> long on pullback to 6h Kijun-sen
            if price_above_kumo and price_below_kijun and vol_confirmed:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            # Strong downtrend: price below weekly Kumo -> short on rally to 6h Kijun-sen
            elif price_below_kumo and price_above_kijun and vol_confirmed:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            # Ranging: price inside weekly Kumo -> fade at Kumo edges
            elif price_in_kumo:
                # Long near Kumo bottom (support)
                if close[i] <= kumo_bottom[i] * 1.001 and vol_confirmed:  # within 0.1% of Kumo bottom
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short near Kumo top (resistance)
                elif close[i] >= kumo_top[i] * 0.999 and vol_confirmed:   # within 0.1% of Kumo top
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals