#!/usr/bin/env python3
"""
Hypothesis: 1h primary with 4h HMA trend filter + RSI pullback entries + Bollinger regime detection.
- 4h HMA(21) determines bull/bear regime (call get_htf_data ONCE before loop)
- 1h RSI(14) for pullback entries (buy when RSI<45 in uptrend, sell when RSI>55 in downtrend)
- Bollinger BandWidth percentile for regime: squeeze = reduce size, expansion = full size
- ATR(14) stoploss: exit when price moves 2.5*ATR against position
- Discrete sizing: 0.0, ±0.20, ±0.35 to minimize fee churn
Why this might work: MTF trend filter avoids whipsaws, RSI pullback gets better entries,
regime detection reduces size in choppy markets. Learned from #001/#002 failures.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_bb_regime_1h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average for smoother trend detection."""
    close_s = pd.Series(close)
    wma1 = close_s.rolling(window=period//2, min_periods=period//2).mean()
    wma2 = close_s.rolling(window=period, min_periods=period).mean()
    wma_diff = 2 * wma1 - wma2
    hma = wma_diff.rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility and stoploss."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index for entry timing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi

def calculate_bollinger_bandwidth(close, period=20, std_mult=2.0):
    """Bollinger BandWidth for regime detection (squeeze vs expansion)."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return bandwidth

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === MTF: Load 4h data ONCE before loop (CRITICAL RULE #1) ===
    df_4h = get_htf_data(prices, '4h')
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Also get 12h for additional trend confirmation
    df_12h = get_htf_data(prices, '12h')
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # === 1h Indicators (computed once, vectorized) ===
    rsi = calculate_rsi(close, 14)
    atr = calculate_atr(high, low, close, 14)
    bb_width = calculate_bollinger_bandwidth(close, 20, 2.0)
    
    # BB Width percentile for regime (rolling 100-bar lookback)
    bb_percentile = np.zeros(n)
    for i in range(100, n):
        bb_percentile[i] = np.percentile(bb_width[max(0,i-100):i+1], 50)
    
    # Current BB width vs its median (squeeze = low, expansion = high)
    bb_regime = bb_width / (bb_percentile + 1e-9)
    
    # === Signal Generation ===
    signals = np.zeros(n)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.20
    prev_signal = 0.0
    entry_price = 0.0
    position_side = 0
    
    for i in range(100, n):
        # HTF trend signals
        trend_4h = 1.0 if hma_4h_aligned[i] > hma_4h_aligned[i-10] else -1.0
        trend_12h = 1.0 if hma_12h_aligned[i] > hma_12h_aligned[i-10] else -1.0
        
        # Combined HTF trend (both must agree for strong signal)
        htf_trend = 0.0
        if trend_4h == 1.0 and trend_12h == 1.0:
            htf_trend = 1.0
        elif trend_4h == -1.0 and trend_12h == -1.0:
            htf_trend = -1.0
        elif trend_4h == 1.0 or trend_12h == 1.0:
            htf_trend = 0.5  # weak bull
        elif trend_4h == -1.0 or trend_12h == -1.0:
            htf_trend = -0.5  # weak bear
        
        # Regime filter (BB Width)
        size_mult = 1.0
        if bb_regime[i] < 0.7:  # squeeze - reduce size
            size_mult = 0.5
        elif bb_regime[i] > 1.5:  # expansion - full size
            size_mult = 1.0
        else:
            size_mult = 0.75
        
        # Current position tracking
        if position_side != 0 and signals[i-1] != 0:
            entry_price = np.where(position_side > 0, 
                                   np.where(entry_price == 0, close[i-1], entry_price),
                                   entry_price)
            entry_price = np.where(position_side < 0,
                                   np.where(entry_price == 0, close[i-1], entry_price),
                                   entry_price)
        
        # === Entry Logic ===
        signal = 0.0
        
        if htf_trend >= 0.5:  # Bullish HTF
            if rsi[i] < 45:  # Pullback entry
                signal = SIZE_FULL * size_mult
            elif rsi[i] > 70:  # Overbought - reduce or exit
                signal = 0.0
            elif prev_signal > 0:
                signal = prev_signal  # hold
        elif htf_trend <= -0.5:  # Bearish HTF
            if rsi[i] > 55:  # Pullback entry for short
                signal = -SIZE_FULL * size_mult
            elif rsi[i] < 30:  # Oversold - reduce or exit
                signal = 0.0
            elif prev_signal < 0:
                signal = prev_signal  # hold
        
        # === Stoploss Logic (CRITICAL RULE #6) ===
        if position_side > 0 and prev_signal > 0:
            if close[i] < entry_price - 2.5 * atr[i]:
                signal = 0.0  # Long stoploss hit
        elif position_side < 0 and prev_signal < 0:
            if close[i] > entry_price + 2.5 * atr[i]:
                signal = 0.0  # Short stoploss hit
        
        # Update position tracking
        if signal != 0 and prev_signal == 0:
            position_side = np.sign(signal)
            entry_price = close[i]
        elif signal == 0 and prev_signal != 0:
            position_side = 0
            entry_price = 0
        
        # Discretize signal to reduce churn
        if abs(signal) < 0.10:
            signal = 0.0
        elif signal > 0:
            signal = SIZE_HALF if signal < SIZE_FULL * 0.8 else SIZE_FULL
        else:
            signal = -SIZE_HALF if signal > -SIZE_FULL * 0.8 else -SIZE_FULL
        
        signals[i] = signal
        prev_signal = signal
    
    return signals