#!/usr/bin/env python3
"""
Experiment #8035: 6-hour Ichimoku Cloud with 1-week trend filter and volume confirmation.
Hypothesis: Price breaking above/below the Ichimoku Cloud on 6h with volume >2x 20-period MA 
and aligned weekly trend (price above/below weekly Kumo cloud) captures sustained moves 
with appropriate frequency in both bull and bear markets. The weekly timeframe provides 
strong trend context to reduce whipsaw while the Ichimoku Cloud acts as dynamic support/resistance.
Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8035_6h_ichimoku_1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENkou_B_PERIOD = 52
SENkou_SHIFT = 26
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku Cloud components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    tenkan_sen = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() + 
                  pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    kijun_sen = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() + 
                 pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(SENkou_SHIFT)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    senkou_span_b = ((pd.Series(high).rolling(window=SENkou_B_PERIOD, min_periods=SENkou_B_PERIOD).max() + 
                      pd.Series(low).rolling(window=SENkou_B_PERIOD, min_periods=SENkou_B_PERIOD).min()) / 2).shift(SENkou_SHIFT)
    
    # Chikou Span (Lagging Span): Close shifted 26 periods behind
    chikou_span = pd.Series(close).shift(-SENkou_SHIFT)
    
    return tenkan_sen.values, kijun_sen.values, senkou_span_a.values, senkou_span_b.values, chikou_span.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w Ichimoku Cloud
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tenkan_1w, kijun_1w, senkou_a_1w, senkou_b_1w, chikou_1w = calculate_ichimoku(high_1w, low_1w, close_1w)
    
    # Determine if price is above or below the Kumo (cloud)
    # Kumo top = max(senkou_a, senkou_b), Kumo bottom = min(senkou_a, senkou_b)
    kumo_top = np.maximum(senkou_a_1w, senkou_b_1w)
    kumo_bottom = np.minimum(senkou_a_1w, senkou_b_1w)
    
    # Price above cloud = bullish, below cloud = bearish
    price_vs_kumo = np.where(close_1w > kumo_top, 1, 
                            np.where(close_1w < kumo_bottom, -1, 0))  # 1=bullish, -1=bearish, 0=in cloud
    price_vs_kumo_aligned = align_htf_to_ltf(prices, df_1w, price_vs_kumo)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku Cloud on 6h
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # Kumo (cloud) on 6h
    kumo_top_6h = np.maximum(senkou_a, senkou_b)
    kumo_bottom_6h = np.minimum(senkou_a, senkou_b)
    
    # Price above/below cloud
    price_above_kumo = close > kumo_top_6h
    price_below_kumo = close < kumo_bottom_6h
    
    # TK Cross (Tenkan/Kijun crossover)
    tk_cross_up = (tenkan > kijun) & (np.roll(tenkan, 1) <= np.roll(kijun, 1))
    tk_cross_down = (tenkan < kijun) & (np.roll(tenkan, 1) >= np.roll(kijun, 1))
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
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
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENkou_B_PERIOD, SENkou_SHIFT, 
                VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(price_vs_kumo_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend bias
        bull_bias = price_vs_kumo_aligned[i] == 1   # Weekly price above Kumo
        bear_bias = price_vs_kumo_aligned[i] == -1  # Weekly price below Kumo
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions - TK cross in direction of weekly trend with price outside cloud
        long_entry = bull_bias and tk_cross_up[i] and price_above_kumo[i] and volume_confirmed
        short_entry = bear_bias and tk_cross_down[i] and price_below_kumo[i] and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals
</lyzard>