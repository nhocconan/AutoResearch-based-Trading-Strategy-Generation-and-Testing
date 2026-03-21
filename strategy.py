#!/usr/bin/env python3
"""
Hypothesis: 4h primary with 1d KAMA trend + BB regime detection
- 1d KAMA provides macro trend bias (slower, more stable than 4h)
- 4h KAMA for entry timing (adaptive to volatility)
- BB Width percentile detects trend vs mean reversion regime
- Asymmetric sizing based on HTF trend strength
- ATR trailing stop for risk management
- Fewer trades on 4h = less fee drag
Timeframe: 4h (primary), 1d (HTF filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_bb_regime_4h_v2"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i >= er_period:
            change = abs(close[i] - close[i - er_period])
            noise = np.sum(np.abs(np.diff(close[max(0, i - er_period):i + 1])))
            er = change / (noise + 1e-10) if noise > 0 else 0
        else:
            er = 0
        
        fast_sc = 2 / (fast_period + 1)
        slow_sc = 2 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    return atr

def calculate_bb_width(close, period=20, num_std=2):
    """Bollinger Band Width for regime detection"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + num_std * std
    lower = sma - num_std * std
    bb_width = (upper - lower) / (sma + 1e-10)
    return bb_width, sma, std

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for macro trend
    kama_1d_21 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_50 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=50)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    kama_1d_50_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_50)
    
    # 4h indicators
    kama_4h_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_4h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    atr = calculate_atr(high, low, close, period=14)
    bb_width, bb_sma, bb_std = calculate_bb_width(close, period=20, num_std=2)
    
    # BB Width percentile for regime detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x) if len(x) > 0 else 0.5, raw=False
    ).values
    bb_width_percentile = np.nan_to_num(bb_width_percentile, 0.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend regime (1d KAMA)
        htf_bull = kama_1d_21_aligned[i] > kama_1d_50_aligned[i] and close[i] > kama_1d_21_aligned[i]
        htf_bear = kama_1d_21_aligned[i] < kama_1d_50_aligned[i] and close[i] < kama_1d_21_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # 1d trend strength
        if i >= 4 and kama_1d_21_aligned[i-4] > 0:
            kama_slope_1d = (kama_1d_21_aligned[i] - kama_1d_21_aligned[i-4]) / kama_1d_21_aligned[i-4]
        else:
            kama_slope_1d = 0
        htf_strength = min(abs(kama_slope_1d) * 100, 2.0)
        
        # 4h trend
        trend_4h = 1.0 if kama_4h_21[i] > kama_4h_50[i] else -1.0
        
        # BB regime: low width = squeeze (trend coming), high width = expansion (trend active)
        bb_squeeze = bb_width_percentile[i] < 0.3
        bb_expansion = bb_width_percentile[i] > 0.7
        
        # ATR stoploss level
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first
        if position_side == 1:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - atr_stop
            if close[i] < trailing_stop or close[i] < entry_price - atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        elif position_side == -1:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + atr_stop
            if close[i] > trailing_stop or close[i] > entry_price + atr_stop:
                signals[i] = 0.0
                position_side = 0
                prev_signal = 0.0
                continue
        
        # Determine position size based on HTF strength
        if htf_bull or htf_bear:
            size = min(SIZE_BASE + htf_strength * 0.05, SIZE_MAX)
        else:
            size = SIZE_BASE * 0.5
        
        # Entry logic - regime adaptive
        if htf_bull:  # Bull regime - prefer longs
            # Trend continuation on 4h
            if trend_4h > 0 and close[i] > kama_4h_21[i]:
                signals[i] = size
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # Pullback entry
            elif trend_4h > 0 and close[i] < kama_4h_21[i] and close[i] > kama_4h_50[i]:
                signals[i] = size * 0.8
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # Overbought exit
            elif close[i] > kama_4h_21[i] * 1.05:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
                
        elif htf_bear:  # Bear regime - prefer shorts
            # Trend continuation on 4h
            if trend_4h < 0 and close[i] < kama_4h_21[i]:
                signals[i] = -size
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # Pullback entry
            elif trend_4h < 0 and close[i] > kama_4h_21[i] and close[i] < kama_4h_50[i]:
                signals[i] = -size * 0.8
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # Oversold exit
            elif close[i] < kama_4h_21[i] * 0.95:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
                
        else:  # Neutral regime - mean reversion only
            if bb_squeeze:
                # Wait for breakout - stay flat or hold existing
                signals[i] = prev_signal
            else:
                # Mean reversion around KAMA
                if close[i] > kama_4h_21[i] * 1.02:
                    signals[i] = -size * 0.5
                    if prev_signal == 0:
                        position_side = -1
                        entry_price = close[i]
                        lowest_since_entry = close[i]
                elif close[i] < kama_4h_21[i] * 0.98:
                    signals[i] = size * 0.5
                    if prev_signal == 0:
                        position_side = 1
                        entry_price = close[i]
                        highest_since_entry = close[i]
                else:
                    signals[i] = 0.0
                    position_side = 0
        
        # Discretize signal to reduce churn
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif signals[i] > 0:
            signals[i] = min(max(round(signals[i] / 0.05) * 0.05, 0.15), SIZE_MAX)
        else:
            signals[i] = max(min(round(signals[i] / 0.05) * 0.05, -0.15), -SIZE_MAX)
        
        prev_signal = signals[i]
    
    return signals