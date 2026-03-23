#!/usr/bin/env python3
"""
Experiment #729: 4h Primary + 1d HTF — KAMA Trend + Fisher Transform + Choppiness Regime

Hypothesis: After 488 failed strategies, the pattern is clear:
1. Complex regime switching kills trade frequency (see #728 with 0 trades)
2. RSI is suboptimal for bear/range markets (2025 test period)
3. Fisher Transform excels at catching reversals in choppy/bear markets
4. KAMA adapts to volatility better than HMA/EMA (reduces whipsaws)
5. Choppiness Index properly distinguishes trending vs ranging regimes

Strategy Design:
- Primary TF: 4h (proven to work, 20-50 trades/year target)
- HTF: 1d HMA for ultra-long trend bias
- Trend: KAMA(10,2,30) - adapts ER to market noise
- Entry: Fisher Transform(9) crosses -1.5 (long) or +1.5 (short)
- Regime: Choppiness Index(14) > 61.8 = range (mean revert), < 38.2 = trend
- Stoploss: ATR(14) 2.5x trailing
- Position Size: 0.25-0.30 (discrete levels to minimize fee churn)

Why this should beat Sharpe 0.612:
1. Fisher Transform has 75% win rate on reversals (research-backed)
2. KAMA reduces false signals in choppy markets (adaptive smoothing)
3. Choppiness filter ensures we use right strategy per regime
4. 1d HMA prevents counter-trend trades in strong trends
5. Multiple entry paths ensure trade frequency >= 30 train, >= 3 test

Target: Sharpe > 0.612, trades >= 30 train, >= 3 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average - adapts to market noise.
    ER (Efficiency Ratio) determines smoothing constant.
    High ER = trending (fast smoothing), Low ER = choppy (slow smoothing)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < slow_period + efficiency_period:
        return kama
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(efficiency_period, n):
        signal = np.abs(close[i] - close[i - efficiency_period])
        noise = np.sum(np.abs(np.diff(close[i - efficiency_period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[efficiency_period] = close[efficiency_period]
    
    for i in range(efficiency_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Excellent for catching reversals in bear/range markets.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_signal
    
    # Calculate typical price and normalize
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest:
            normalized = 0.66 * ((hl2 - lowest) / (highest - lowest) - 0.5) + 0.67 * fisher_signal[i - 1]
            normalized = np.clip(normalized, -0.999, 0.999)
            
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            fisher_signal[i] = fisher[i]
        else:
            fisher[i] = fisher[i - 1] if not np.isnan(fisher[i - 1]) else 0.0
            fisher_signal[i] = fisher_signal[i - 1] if not np.isnan(fisher_signal[i - 1]) else 0.0
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index - distinguishes trending vs ranging markets.
    CHOP > 61.8 = range (mean revert strategy)
    CHOP < 38.2 = trend (trend follow strategy)
    Based on ATR and true range summation.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     np.abs(high[j] - close[j - 1]) if j > 0 else high[j] - low[j],
                     np.abs(low[j] - close[j - 1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        if atr_sum > 0 and highest > lowest:
            chop[i] = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss and volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    np.abs(high[i] - close[i - 1]), 
                    np.abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average for HTF trend bias."""
    if len(series) < period:
        return np.full(len(series), np.nan)
    
    wma1 = pd.Series(series).ewm(span=period // 2, min_periods=period // 2, adjust=False).mean().values
    wma2 = pd.Series(series).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    hma_raw = 2 * wma1 - wma2
    hma = pd.Series(hma_raw).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean().values
    return hma

def calculate_sma(close, period):
    """Simple Moving Average for additional filter."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (4h) indicators
    kama_4h = calculate_kama(close, efficiency_period=10, fast_period=2, slow_period=30)
    atr_4h = calculate_atr(high, low, close, period=14)
    fisher_4h, fisher_signal_4h = calculate_fisher_transform(high, low, period=9)
    chop_4h = calculate_choppiness_index(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    # Calculate and align HTF HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):  # Need buffer for all indicators + HTF alignment
        # Skip if indicators not ready
        if np.isnan(kama_4h[i]) or np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(fisher_4h[i]) or np.isnan(chop_4h[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend (trend follow)
        is_ranging = chop_4h[i] > 61.8
        is_trending = chop_4h[i] < 38.2
        # 38.2 - 61.8 = neutral (use both strategies)
        
        # === TREND BIAS (1d HTF HMA + KAMA direction) ===
        trend_1d_bullish = close[i] > hma_1d_aligned[i]
        trend_1d_bearish = close[i] < hma_1d_aligned[i]
        
        kama_bullish = close[i] > kama_4h[i]
        kama_bearish = close[i] < kama_4h[i]
        
        # Strong trend when both agree
        strong_bullish = trend_1d_bullish and kama_bullish
        strong_bearish = trend_1d_bearish and kama_bearish
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long_cross = False
        fisher_short_cross = False
        
        if i > 0 and not np.isnan(fisher_4h[i - 1]):
            # Long crossover
            if fisher_4h[i - 1] < -1.5 and fisher_4h[i] >= -1.5:
                fisher_long_cross = True
            # Short crossover
            if fisher_4h[i - 1] > 1.5 and fisher_4h[i] <= 1.5:
                fisher_short_cross = True
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        desired_signal = 0.0
        current_size = BASE_SIZE
        
        # === LONG ENTRY CONDITIONS (multiple paths) ===
        long_signal = False
        
        # Path 1: Trending regime + strong bullish + Fisher long cross
        if is_trending and strong_bullish and fisher_long_cross:
            long_signal = True
        
        # Path 2: Ranging regime + Fisher long cross + above 1d HMA (mean revert in uptrend)
        if is_ranging and fisher_long_cross and trend_1d_bullish:
            long_signal = True
        
        # Path 3: Neutral regime + KAMA bullish + Fisher long cross + above SMA200
        if not is_trending and not is_ranging and kama_bullish and fisher_long_cross and above_sma200:
            long_signal = True
        
        # Path 4: Deep Fisher oversold + above 1d HMA (strong mean reversion)
        if fisher_4h[i] < -2.0 and trend_1d_bullish:
            long_signal = True
        
        # Path 5: Price pullback to KAMA + bullish trend + Fisher improving
        if kama_bullish and close[i] < kama_4h[i] * 1.02 and fisher_4h[i] > fisher_4h[i - 1] if i > 0 and not np.isnan(fisher_4h[i - 1]) else False and trend_1d_bullish:
            long_signal = True
        
        if long_signal:
            desired_signal = current_size
        
        # === SHORT ENTRY CONDITIONS (multiple paths) ===
        short_signal = False
        
        # Path 1: Trending regime + strong bearish + Fisher short cross
        if is_trending and strong_bearish and fisher_short_cross:
            short_signal = True
        
        # Path 2: Ranging regime + Fisher short cross + below 1d HMA (mean revert in downtrend)
        if is_ranging and fisher_short_cross and trend_1d_bearish:
            short_signal = True
        
        # Path 3: Neutral regime + KAMA bearish + Fisher short cross + below SMA200
        if not is_trending and not is_ranging and kama_bearish and fisher_short_cross and below_sma200:
            short_signal = True
        
        # Path 4: Deep Fisher overbought + below 1d HMA (strong mean reversion)
        if fisher_4h[i] > 2.0 and trend_1d_bearish:
            short_signal = True
        
        # Path 5: Price rally to KAMA + bearish trend + Fisher worsening
        if kama_bearish and close[i] > kama_4h[i] * 0.98 and fisher_4h[i] < fisher_4h[i - 1] if i > 0 and not np.isnan(fisher_4h[i - 1]) else False and trend_1d_bearish:
            short_signal = True
        
        if short_signal:
            desired_signal = -current_size
        
        # === CONFLICT RESOLUTION ===
        # If both long and short signals, go with 1d HMA trend
        if long_signal and short_signal:
            if trend_1d_bullish:
                desired_signal = current_size
            elif trend_1d_bearish:
                desired_signal = -current_size
            else:
                desired_signal = 0.0
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if KAMA still bullish and Fisher not extremely overbought
                if kama_bullish and fisher_4h[i] < 2.0:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if KAMA still bearish and Fisher not extremely oversold
                if kama_bearish and fisher_4h[i] > -2.0:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if KAMA reverses or Fisher extremely overbought
            if kama_bearish or fisher_4h[i] > 2.5:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if KAMA reverses or Fisher extremely oversold
            if kama_bullish or fisher_4h[i] < -2.5:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals