#!/usr/bin/env python3
"""
Hypothesis: 12h primary with 1d KAMA trend + BB Width regime + RSI timing
- 12h timeframe reduces noise vs 15m/1h, fewer false signals
- KAMA adapts to market efficiency (slow in chop, fast in trend)
- 1d KAMA provides stable HTF trend filter
- BB Width percentile detects regime (squeeze=range, expand=trend)
- RSI(14) for entry timing within trend direction
- ATR trailing stop for risk management
- Asymmetric sizing: 0.30 in strong trend, 0.15 in weak/uncertain
Timeframe: 12h (primary), 1d (HTF trend filter)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_bb_regime_12h_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    kama = np.zeros(n)
    
    # Change = absolute price change over er_period
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    # Volatility = sum of absolute single-period changes
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1))[1:])
    
    # Efficiency Ratio
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er[:er_period] = 0
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

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
    upper = sma + 2 * std
    lower = sma - 2 * std
    bb_width = (upper - lower) / sma
    return bb_width, sma, std

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Percentile of BB Width vs recent history"""
    n = len(bb_width)
    percentile = np.zeros(n)
    
    for i in range(lookback, n):
        window = bb_width[i-lookback:i+1]
        percentile[i] = np.sum(window < bb_width[i]) / len(window)
    
    percentile[:lookback] = 0.5
    return percentile

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # 1d KAMA for HTF trend direction
    kama_1d_21 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_50 = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=50)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    kama_1d_50_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_50)
    
    # 12h indicators
    kama_12h_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_12h_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=50)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    bb_width, bb_sma, bb_std = calculate_bb_width(close, period=20)
    bb_width_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30   # Size in confirmed trend
    SIZE_WEAK = 0.15    # Size in weak/uncertain trend
    SIZE_MAX = 0.35
    
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # HTF trend regime (1d KAMA)
        htf_bull = kama_1d_21_aligned[i] > kama_1d_50_aligned[i] and close[i] > kama_1d_21_aligned[i]
        htf_bear = kama_1d_21_aligned[i] < kama_1d_50_aligned[i] and close[i] < kama_1d_21_aligned[i]
        
        # 12h trend
        ltf_bull = kama_12h_21[i] > kama_12h_50[i]
        ltf_bear = kama_12h_21[i] < kama_12h_50[i]
        
        # Regime detection via BB Width percentile
        # Low percentile = squeeze (range), High percentile = expansion (trend)
        regime_trend = bb_width_pct[i] > 0.6  # Expanding bands = trending
        regime_range = bb_width_pct[i] < 0.4  # Squeezing bands = ranging
        
        # KAMA slope for trend strength
        kama_slope = (kama_12h_21[i] - kama_12h_21[i-5]) / (kama_12h_21[i-5] + 1e-10)
        trend_strong = abs(kama_slope) > 0.005
        
        # ATR stoploss level
        atr_stop = 2.5 * atr[i]
        
        # Check stoploss first - MUST exit on stop
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
        
        # Determine position size based on regime
        if regime_trend and trend_strong:
            size = SIZE_TREND
        else:
            size = SIZE_WEAK
        
        # Entry logic - simpler to ensure trades are generated
        if htf_bull and ltf_bull:  # Strong bull alignment
            # RSI pullback entry
            if rsi[i] < 55 and rsi[i] > 35:
                signals[i] = size
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            # RSI overbought - exit
            elif rsi[i] > 70:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
                
        elif htf_bear and ltf_bear:  # Strong bear alignment
            # RSI pullback entry
            if rsi[i] > 45 and rsi[i] < 65:
                signals[i] = -size
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            # RSI oversold - exit
            elif rsi[i] < 30:
                signals[i] = 0.0
                position_side = 0
            else:
                signals[i] = prev_signal
                
        elif regime_range:  # Range-bound - mean reversion
            if rsi[i] < 30:
                signals[i] = size * 0.5
                if prev_signal == 0:
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = close[i]
            elif rsi[i] > 70:
                signals[i] = -size * 0.5
                if prev_signal == 0:
                    position_side = -1
                    entry_price = close[i]
                    lowest_since_entry = close[i]
            else:
                signals[i] = prev_signal
        
        else:  # Uncertain regime - reduce or flat
            if prev_signal != 0:
                signals[i] = prev_signal * 0.5  # Reduce position
            else:
                signals[i] = 0.0
        
        # Discretize signal to reduce churn (every change costs 0.10% fees)
        if abs(signals[i]) < 0.10:
            signals[i] = 0.0
        elif signals[i] > 0:
            signals[i] = min(max(round(signals[i] / 0.05) * 0.05, 0.15), SIZE_MAX)
        else:
            signals[i] = max(min(round(signals[i] / 0.05) * 0.05, -0.15), -SIZE_MAX)
        
        prev_signal = signals[i]
    
    return signals