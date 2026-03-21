#!/usr/bin/env python3
"""
Hypothesis: 1h primary with 4h HMA trend + KAMA adaptive entry + BB regime filter
- 4h HMA(21) provides stable trend direction (proven in keeper strategy #004)
- 1h KAMA(10) adapts to volatility - faster in trends, slower in chop
- BB Width percentile detects squeeze (mean reversion) vs expansion (trend)
- RSI(14) for pullback entries - lenient thresholds to ensure trades
- Asymmetric: larger longs in bull, smaller shorts in bear (BTC 2025 is bearish)
- ATR(14) stoploss at 2.5x for risk management
- Discrete sizing: 0.0, ±0.20, ±0.30 to minimize fee churn
Timeframe: 1h (primary), 4h (HTF trend filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_hma_bb_regime_1h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market noise"""
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        # Efficiency Ratio
        if i >= er_period:
            signal = abs(close[i] - close[i - er_period])
            noise = sum(abs(close[j] - close[j-1]) for j in range(i - er_period + 1, i + 1))
            er = signal / (noise + 1e-10) if noise > 0 else 0
        else:
            er = 0.5
        
        # Smoothing constant
        sc = (er * (2/(fast_period+1) - 2/(slow_period+1)) + 2/(slow_period+1)) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
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

def calculate_bb_width(close, period=20):
    """Bollinger Band Width for regime detection"""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    bb_width = (bb_upper - bb_lower) / (sma + 1e-10)
    return bb_width, sma, std

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    # 4h HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_50 = calculate_hma(df_4h['close'].values, 50)
    hma_4h_21_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    hma_4h_50_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_50)
    
    # 1h indicators
    kama_1h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=20)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, 14)
    bb_width, bb_sma, bb_std = calculate_bb_width(close, 20)
    
    # BB Width percentile for regime
    bb_width_percentile = pd.Series(bb_width).rolling(window=100, min_periods=50).apply(
        lambda x: np.sum(x < x.iloc[-1]) / len(x) if len(x) > 0 else 0.5, raw=False
    ).values
    bb_width_percentile = np.nan_to_num(bb_width_percentile, 0.5)
    
    signals = np.zeros(n)
    SIZE_LONG = 0.30   # 30% for longs
    SIZE_SHORT = 0.20  # 20% for shorts (asymmetric - bearish 2025)
    SIZE_HALF = 0.15   # Half position for take profit
    
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend regime (4h HMA)
        htf_bull = hma_4h_21_aligned[i] > hma_4h_50_aligned[i] and close[i] > hma_4h_21_aligned[i]
        htf_bear = hma_4h_21_aligned[i] < hma_4h_50_aligned[i] and close[i] < hma_4h_21_aligned[i]
        
        # 4h trend slope
        hma_slope = (hma_4h_21_aligned[i] - hma_4h_21_aligned[max(0, i-8)]) / (hma_4h_21_aligned[max(0, i-8)] + 1e-10)
        
        # 1h trend (KAMA crossover)
        trend_1h = 1.0 if kama_fast[i] > kama_1h[i] else -1.0
        
        # BB regime: low width = squeeze (mean reversion), high width = trend
        bb_squeeze = bb_width_percentile[i] < 0.3
        bb_expansion = bb_width_percentile[i] > 0.7
        
        # RSI levels
        rsi_oversold = rsi[i] < 40
        rsi_overbought = rsi[i] > 60
        rsi_neutral = 40 <= rsi[i] <= 60
        
        # ATR stoploss
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first (MANDATORY)
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
        
        # Entry logic - simplified to ensure trades happen
        if htf_bull:
            # Bull regime: prefer longs
            if trend_1h > 0 and rsi_oversold:
                # Pullback entry in uptrend
                signals[i] = SIZE_LONG
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            elif trend_1h > 0 and rsi_neutral:
                # Continuation entry
                signals[i] = SIZE_LONG
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            elif rsi_overbought and position_side == 1:
                # Take profit on overbought
                signals[i] = SIZE_HALF
            else:
                signals[i] = prev_signal
                
        elif htf_bear:
            # Bear regime: smaller shorts
            if trend_1h < 0 and rsi_overbought:
                # Pullback entry in downtrend
                signals[i] = -SIZE_SHORT
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            elif trend_1h < 0 and rsi_neutral:
                # Continuation entry
                signals[i] = -SIZE_SHORT
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            elif rsi_oversold and position_side == -1:
                # Take profit on oversold
                signals[i] = -SIZE_HALF
            else:
                signals[i] = prev_signal
        else:
            # Neutral regime: mean reversion only (BB squeeze)
            if bb_squeeze:
                if rsi_oversold:
                    signals[i] = SIZE_LONG * 0.5
                    if prev_signal == 0:
                        position_side = 1
                        entry_price = close[i]
                        highest_since_entry = close[i]
                elif rsi_overbought:
                    signals[i] = -SIZE_SHORT * 0.5
                    if prev_signal == 0:
                        position_side = -1
                        entry_price = close[i]
                        lowest_since_entry = close[i]
                else:
                    signals[i] = 0.0
                    position_side = 0
            else:
                signals[i] = prev_signal
        
        # Discretize signal to reduce churn (CRITICAL for fees)
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif signals[i] > 0:
            if signals[i] >= 0.25:
                signals[i] = SIZE_LONG
            else:
                signals[i] = SIZE_HALF
        else:
            if signals[i] <= -0.25:
                signals[i] = -SIZE_SHORT
            else:
                signals[i] = -SIZE_HALF
        
        prev_signal = signals[i]
    
    return signals