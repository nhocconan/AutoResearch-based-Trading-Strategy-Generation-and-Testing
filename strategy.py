#!/usr/bin/env python3
"""
Experiment #314: 1d Primary + 1w HTF — KAMA Trend + Fisher Transform + Choppiness Regime v1

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance for BTC/ETH.
KAMA adapts to market volatility better than EMA/HMA. Fisher Transform catches reversals
in bear/range markets (2025 test period). Choppiness Index switches between mean-reversion
and trend-following modes.

Key design:
1. KAMA(10) for adaptive trend - responds faster in trends, slower in chop
2. Fisher Transform(9) for entry timing - crosses at -1.0/+1.0 levels (looser for more trades)
3. Choppiness(14) regime: >55 = chop (mean revert), <45 = trend (breakout)
4. 1w HMA(21) for major trend bias - only trade in direction of weekly trend
5. ATR(14) stoploss at 2.5x from entry

Position sizing: 0.25 base, 0.30 when 1w aligned
Target: 20-50 trades/year, Sharpe>0.40, DD>-40%

CRITICAL: All indicators use min_periods. No look-ahead. get_htf_data() called ONCE.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_fisher_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts smoothing based on market efficiency ratio
    """
    n = len(close)
    if n < slow_period + 5:
        return np.full(n, np.nan)
    
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(slow_period, n):
        price_change = abs(close[i] - close[i-slow_period])
        volatility = np.sum(np.abs(np.diff(close[i-slow_period:i+1])))
        if volatility > 1e-10:
            er[i] = price_change / volatility
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[slow_period] = close[slow_period]
    
    for i in range(slow_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for better reversal detection
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    trigger = np.zeros(n)
    trigger[:] = np.nan
    
    for i in range(period-1, n):
        # Calculate typical price
        hl2 = (high[i-period+1:i+1] + low[i-period+1:i+1]) / 2
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (hl2[-1] - lowest) / price_range - 1.0
        normalized = max(-0.999, min(0.999, normalized))  # Clamp for log
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i >= period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index
    CHOP > 61.8 = choppy/range bound
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    kama = calculate_kama(close, period=10)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(kama[i]) or np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION ===
        choppy_threshold = 55.0
        trending_threshold = 45.0
        
        is_choppy = chop[i] > choppy_threshold
        is_trending = chop[i] < trending_threshold
        
        # === WEEKLY TREND BIAS ===
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === KAMA TREND ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === FISHER TRANSFORM SIGNALS (LOOSENED for more trades) ===
        fisher_long = False
        fisher_short = False
        
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_trigger[i]):
            # Long: Fisher crosses above -1.0 from below (oversold reversal)
            fisher_long = fisher_trigger[i] < -1.0 and fisher[i] > fisher_trigger[i]
            # Short: Fisher crosses below +1.0 from above (overbought reversal)
            fisher_short = fisher_trigger[i] > 1.0 and fisher[i] < fisher_trigger[i]
        
        # === KAMA CROSSOVER (additional entry signal for more trades) ===
        kama_cross_long = False
        kama_cross_short = False
        if i > 0 and not np.isnan(kama[i-1]):
            kama_cross_long = close[i-1] <= kama[i-1] and close[i] > kama[i]
            kama_cross_short = close[i-1] >= kama[i-1] and close[i] < kama[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if is_choppy:
            # Mean reversion mode: trade Fisher reversals with weekly alignment
            # Long when oversold + weekly support
            if fisher_long and weekly_bull:
                desired_signal = SIZE_STRONG
            # Short when overbought + weekly resistance
            elif fisher_short and weekly_bear:
                desired_signal = -SIZE_STRONG
            # Fallback: KAMA cross in choppy market (mean revert to KAMA)
            elif kama_cross_long and weekly_bull:
                desired_signal = SIZE_BASE
            elif kama_cross_short and weekly_bear:
                desired_signal = -SIZE_BASE
        
        elif is_trending:
            # Trend following mode: trade with KAMA + weekly alignment
            # Long: KAMA bull + weekly bull
            if kama_bull and weekly_bull:
                desired_signal = SIZE_BASE
            # Short: KAMA bear + weekly bear
            elif kama_bear and weekly_bear:
                desired_signal = -SIZE_BASE
            # Add Fisher confirmation for stronger signal
            if fisher_long and kama_bull and weekly_bull:
                desired_signal = SIZE_STRONG
            elif fisher_short and kama_bear and weekly_bear:
                desired_signal = -SIZE_STRONG
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals