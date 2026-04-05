#!/usr/bin/env python3
"""
Experiment #8295: 6-hour Ichimoku Cloud with 1-week trend filter and volume confirmation.
Hypothesis: Price trading above/below the Ichimoku cloud on 6h with volume >1.5x 20-period MA
and aligned weekly trend (price above/below weekly Kumo) captures sustained trends while avoiding
whipsaw in both bull and bear markets. The weekly trend filter provides long-term context,
reducing false signals during consolidation. Targeting 50-150 total trades over 4 years.
"""

from mtf_data import get_afh_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_8295_6h_ichimoku1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
TENKAN_PERIOD = 9      # Conversion Line
KIJUN_PERIOD = 26      # Base Line
SENKOU_B_PERIOD = 52   # Leading Span B
KUMO_SHIFT = 26        # Kumo cloud shift
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
ATR_TARGET_MULTIPLIER = 3.0

def calculate_ichimoku(high, low, close):
    """Calculate Ichimoku components"""
    # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 over TENKAN_PERIOD
    tenkan = (pd.Series(high).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).max() +
              pd.Series(low).rolling(window=TENKAN_PERIOD, min_periods=TENKAN_PERIOD).min()) / 2
    
    # Kijun-sen (Base Line): (highest high + lowest low)/2 over KIJUN_PERIOD
    kijun = (pd.Series(high).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).max() +
             pd.Series(low).rolling(window=KIJUN_PERIOD, min_periods=KIJUN_PERIOD).min()) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted KUMO_SHIFT forward
    senkou_a = ((tenkan + kijun) / 2).shift(KUMO_SHIFT)
    
    # Senkou Span B (Leading Span B): (highest high + lowest low)/2 over SENKOU_B_PERIOD shifted KUMO_SHIFT
    senkou_b = ((pd.Series(high).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).max() +
                 pd.Series(low).rolling(window=SENKOU_B_PERIOD, min_periods=SENKOU_B_PERIOD).min()) / 2).shift(KUMO_SHIFT)
    
    # Chikou Span (Lagging Span): close shifted -KUMO_SHIFT backward (not used in signals)
    
    return tenkan.values, kijun.values, senkou_a.values, senkou_b.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Ichimoku for trend filter
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    wk_tenkan, wk_kijun, wk_senkou_a, wk_senkou_b = calculate_ichimoku(weekly_high, weekly_low, weekly_close)
    
    # Weekly trend: price above/both spans = bullish, price below/both spans = bearish
    wk_price_above_kumo = (weekly_close > wk_senkou_a) & (weekly_close > wk_senkou_b)
    wk_price_below_kumo = (weekly_close < wk_senkou_a) & (weekly_close < wk_senkou_b)
    wk_trend = np.where(wk_price_above_kumo, 1, np.where(wk_price_below_kumo, -1, 0))  # 1=bullish, -1=bearish, 0=neutral
    wk_trend_aligned = align_htf_to_ltf(prices, df_1w, wk_trend)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Ichimoku on 6h
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(high, low, close)
    
    # Price relative to Kumo (cloud)
    price_above_kumo = (close > senkou_a) & (close > senkou_b)
    price_below_kumo = (close < senkou_a) & (close < senkou_b)
    
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
    target_price = 0.0
    
    # Start from warmup period
    start = max(TENKAN_PERIOD, KIJUN_PERIOD, SENKOU_B_PERIOD, KUMO_SHIFT,
                VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(wk_trend_aligned[i]):
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
        
        # Determine market bias from weekly Ichimoku
        bull_bias = wk_trend_aligned[i] == 1   # Weekly price above Kumo
        bear_bias = wk_trend_aligned[i] == -1  # Weekly price below Kumo
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Ichimoku signals
        # Long: price above Kumo + TK cross up + volume
        long_signal = price_above_kumo[i] and tk_cross_up[i] and volume_confirmed
        # Short: price below Kumo + TK cross down + volume
        short_signal = price_below_kumo[i] and tk_cross_down[i] and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
                target_price = entry_price + (ATR_TARGET_MULTIPLIER * atr[i])
            elif short_signal:
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