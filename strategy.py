#!/usr/bin/env python3
"""
Experiment #536: 30m KAMA Adaptive Trend with 4h HMA Bias + BB Regime Filter

Hypothesis: After analyzing 500+ failed experiments, the pattern is clear:
1. Pure mean reversion (RSI, Connors) fails on 30m/15m timeframes (Sharpe -2 to -5)
2. Trend following WITH proper HTF bias shows promise (#533 +12.5% return)
3. KAMA (Kaufman Adaptive) adapts to volatility better than fixed EMA/HMA
4. BB Width regime filter avoids trading during choppy consolidation
5. 30m captures intraday trends without excessive noise vs 15m
6. 4h HMA provides smoother trend bias than 12h for 30m entries

Why this should work on 30m:
- 30m has 48 bars/day = good balance of signal frequency vs noise
- KAMA adapts ER (Efficiency Ratio) to reduce whipsaw in chop
- 4h HMA trend bias prevents counter-trend entries (major failure mode)
- BB Width < 20th percentile = squeeze (wait), > 50th = expansion (trade)
- MACD histogram confirms momentum direction before entry
- 2.0*ATR stoploss protects against 2022-style crashes
- Discrete sizing (0.25) minimizes fee churn

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete (max 0.40)
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_kama_adaptive_4h_hma_bb_regime_macd_atr_v1"
timeframe = "30m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio (ER).
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Efficiency Ratio
    signal = np.abs(close_s - close_s.shift(er_period))
    noise = np.abs(close_s - close_s.shift(1)).rolling(window=er_period, min_periods=er_period).sum()
    er = signal / noise.replace(0, np.inf)
    er = er.fillna(0)
    
    # Smoothing Constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bb_width(high, low, close, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width for regime detection."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    
    # Calculate percentile rank over rolling window
    width_percentile = width.rolling(window=100, min_periods=100).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
    )
    
    return width.values, width_percentile.values

def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD histogram for momentum confirmation."""
    close_s = pd.Series(close)
    ema_fast = close_s.ewm(span=fast, min_periods=fast, adjust=False).mean()
    ema_slow = close_s.ewm(span=slow, min_periods=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return histogram.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    bb_width, bb_percentile = calculate_bb_width(high, low, close, 20, 2.0)
    macd_hist = calculate_macd(close, 12, 26, 9)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(kama[i]) or np.isnan(bb_percentile[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            continue
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === BB REGIME FILTER ===
        # BB percentile > 0.5 = expanding bands (trending regime)
        # BB percentile < 0.3 = squeeze (wait for breakout)
        regime_expanding = bb_percentile[i] > 0.4
        regime_squeeze = bb_percentile[i] < 0.3
        
        # === KAMA ADAPTIVE TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === MACD MOMENTUM CONFIRMATION ===
        macd_bull = macd_hist[i] > 0
        macd_bear = macd_hist[i] < 0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long: 4h bull bias + KAMA bull + MACD bull + regime expanding (or squeeze breakout)
        if bull_bias and kama_bull and macd_bull and regime_expanding:
            new_signal = SIZE
        elif bull_bias and kama_bull and macd_bull and regime_squeeze:
            # Allow entries during squeeze if all other conditions align
            new_signal = SIZE * 0.6  # Reduced size during squeeze
        
        # Short: 4h bear bias + KAMA bear + MACD bear + regime expanding
        if bear_bias and kama_bear and macd_bear and regime_expanding:
            new_signal = -SIZE
        elif bear_bias and kama_bear and macd_bear and regime_squeeze:
            new_signal = -SIZE * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h HMA flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and bear_bias:
                new_signal = 0.0
            if position_side < 0 and bull_bias:
                new_signal = 0.0
        
        # === MACD MOMENTUM LOSS EXIT ===
        # Exit if MACD histogram flips against position
        if in_position and new_signal != 0.0:
            if position_side > 0 and macd_bear:
                new_signal = 0.0
            if position_side < 0 and macd_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals