#!/usr/bin/env python3
"""
Hypothesis: Daily timeframe with weekly trend filter + KAMA adaptive trend + BB regime + RSI timing
- 1d primary provides clean signals with less noise than intraday
- 1w HMA filters long-term trend direction (avoid counter-trend trades)
- KAMA adapts to market efficiency (fast in trends, slow in chop)
- Bollinger Band Width detects regime (squeeze=low vol, expand=trending)
- RSI for entry timing within trend direction
- ATR trailing stop for risk management
- Asymmetric sizing: larger in strong weekly trend, smaller in neutral
Timeframe: 1d (primary), 1w (HTF filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_bb_rsi_regime_1d_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio
        if i >= er_period:
            change = abs(close[i] - close[i - er_period])
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            er = change / (volatility + 1e-10) if volatility > 0 else 0
        else:
            er = 0
        
        # Smoothing constant
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    hull = 2 * wma_half - wma_full
    hma = pd.Series(hull).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Bollinger Bands with width for regime detection"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    pct_b = (close - lower) / (upper - lower + 1e-10)
    return upper, lower, width, pct_b

def calculate_momentum(close, period=10):
    """Rate of Change momentum"""
    roc = np.zeros(len(close))
    roc[period:] = (close[period:] - close[:-period]) / (close[:-period] + 1e-10) * 100
    return roc

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for long-term trend direction
    hma_1w_10 = calculate_hma(df_1w['close'].values, 10)
    hma_1w_20 = calculate_hma(df_1w['close'].values, 20)
    hma_1w_10_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_10)
    hma_1w_20_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_20)
    
    # 1d indicators
    kama_1d_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1d_30 = calculate_kama(close, er_period=10, fast_period=5, slow_period=50)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width, bb_pct_b = calculate_bollinger_bands(close, 20, 2.0)
    momentum = calculate_momentum(close, 10)
    
    # BB Width percentile for regime detection
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x) if len(x) > 0 else 0.5, raw=False
    ).values
    bb_width_percentile = np.nan_to_num(bb_width_percentile, 0.5)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # Base position size 25%
    SIZE_MAX = 0.35   # Max position size 35%
    SIZE_MIN = 0.15   # Min position size 15%
    
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Weekly trend regime
        htf_bull = hma_1w_10_aligned[i] > hma_1w_20_aligned[i] and close[i] > hma_1w_10_aligned[i]
        htf_bear = hma_1w_10_aligned[i] < hma_1w_20_aligned[i] and close[i] < hma_1w_10_aligned[i]
        htf_neutral = not htf_bull and not htf_bear
        
        # Weekly trend strength (slope)
        if i >= 7:
            hma_slope_1w = (hma_1w_10_aligned[i] - hma_1w_10_aligned[i-7]) / (hma_1w_10_aligned[i-7] + 1e-10)
        else:
            hma_slope_1w = 0
        htf_strength = min(abs(hma_slope_1w) * 50, 2.0)  # Cap at 2.0
        
        # Daily trend (KAMA crossover)
        kama_bull = kama_1d_10[i] > kama_1d_30[i]
        kama_bear = kama_1d_10[i] < kama_1d_30[i]
        
        # BB regime
        bb_squeeze = bb_width_percentile[i] < 0.3  # Low volatility
        bb_expand = bb_width_percentile[i] > 0.7   # High volatility / trending
        
        # RSI conditions
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # Momentum confirmation
        mom_positive = momentum[i] > 0
        mom_negative = momentum[i] < 0
        
        # ATR stoploss level
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first (trailing stop)
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
        
        # Determine position size based on HTF strength and regime
        if htf_bull or htf_bear:
            if bb_expand:
                size = min(SIZE_BASE + htf_strength * 0.05, SIZE_MAX)
            else:
                size = SIZE_BASE
        else:
            size = SIZE_MIN  # Reduce size in neutral regime
        
        # Entry logic - asymmetric based on weekly regime
        if htf_bull:  # Bull regime - prefer longs
            # Trend continuation entry
            if kama_bull and rsi_oversold and mom_positive:
                signals[i] = size
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # Pullback entry in BB squeeze
            elif kama_bull and bb_squeeze and rsi[i] < 50:
                signals[i] = size * 0.8
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # Overbought - reduce or exit
            elif rsi_overbought and position_side == 1:
                signals[i] = size * 0.5
            else:
                signals[i] = prev_signal
                
        elif htf_bear:  # Bear regime - prefer shorts
            # Trend continuation entry
            if kama_bear and rsi_overbought and mom_negative:
                signals[i] = -size
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # Pullback entry in BB squeeze
            elif kama_bear and bb_squeeze and rsi[i] > 50:
                signals[i] = -size * 0.8
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # Oversold - reduce or exit
            elif rsi_oversold and position_side == -1:
                signals[i] = -size * 0.5
            else:
                signals[i] = prev_signal
                
        else:  # Neutral regime - mean reversion only
            if rsi_overbought and bb_pct_b[i] > 0.9:
                signals[i] = -size * 0.5
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            elif rsi_oversold and bb_pct_b[i] < 0.1:
                signals[i] = size * 0.5
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            elif rsi_neutral:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
        
        # Discretize signal to reduce churn
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif signals[i] > 0:
            signals[i] = min(max(round(signals[i] / 0.05) * 0.05, SIZE_MIN), SIZE_MAX)
        else:
            signals[i] = max(min(round(signals[i] / 0.05) * 0.05, -SIZE_MIN), -SIZE_MAX)
        
        prev_signal = signals[i]
    
    return signals