#!/usr/bin/env python3
"""
Experiment #832: 12h Primary + 1d HTF — Dual-Regime with Choppiness Filter

Hypothesis: Combining Choppiness Index regime detection with adaptive entry logic
will outperform pure trend or pure mean-reversion strategies. In choppy markets
(CHOP>61.8), use mean-reversion (RSI extremes). In trending markets (CHOP<38.2),
use trend-following (HMA crossover + pullback). 1d HMA provides directional bias.

Key innovations:
1. Choppiness Index(14) for regime detection - switch entry logic dynamically
2. 1d HMA(21) for HTF trend bias - only trade with higher timeframe direction
3. 12h HMA(16/48) for local trend + RSI(14) for pullback timing
4. ATR(14) 2.5x trailing stop for risk management
5. Asymmetric sizing: 0.30 when HTF+LTF align, 0.20 when only LTF signals
6. Loose enough entries to guarantee ≥30 trades/train, ≥3/test

Entry conditions:
- TREND REGIME (CHOP<45): HMA crossover + RSI confirmation + HTF bias
- RANGE REGIME (CHOP>55): RSI extremes (25/75) + HTF bias for direction
- TRANSITION (45-55): Require stronger confluence (both HMA + RSI)

Target: Sharpe>0.50, trades>=30 train, trades>=3 test, DD>-35%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_regime_hma_rsi_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
    
    return chop

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 12h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        is_trend_regime = chop_value < 45.0  # Trending market
        is_range_regime = chop_value > 55.0  # Choppy/range market
        # 45-55 is transition zone - require stronger signals
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA CROSSOVER ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_16[i-1]) and not np.isnan(hma_48[i-1]):
            hma_crossover_long = (hma_16[i-1] <= hma_48[i-1]) and (hma_16[i] > hma_48[i])
            hma_crossover_short = (hma_16[i-1] >= hma_48[i-1]) and (hma_16[i] < hma_48[i])
        
        # === 12h HMA TREND ===
        hma_12h_bull = hma_16[i] > hma_48[i]
        hma_12h_bear = hma_16[i] < hma_48[i]
        
        # === RSI CONDITIONS ===
        rsi_value = rsi_14[i]
        rsi_oversold = rsi_value < 35.0
        rsi_overbought = rsi_value > 65.0
        rsi_extreme_oversold = rsi_value < 25.0
        rsi_extreme_overbought = rsi_value > 75.0
        rsi_neutral = 40.0 < rsi_value < 60.0
        
        # === PRICE VS SMA200 ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        signal_strength = 0  # 0=none, 1=weak, 2=strong
        
        # LONG ENTRIES
        long_conditions = 0
        
        if htf_1d_bull:  # Only long when HTF is bullish
            if is_trend_regime:
                # Trend regime: HMA crossover + RSI confirmation
                if hma_crossover_long:
                    long_conditions += 2
                if hma_12h_bull and rsi_neutral:
                    long_conditions += 1
                if rsi_oversold:
                    long_conditions += 1
            elif is_range_regime:
                # Range regime: RSI mean reversion
                if rsi_extreme_oversold:
                    long_conditions += 2
                elif rsi_oversold:
                    long_conditions += 1
                if price_above_sma200:
                    long_conditions += 1
            else:
                # Transition zone: require multiple confirmations
                if hma_12h_bull and rsi_oversold:
                    long_conditions += 2
                if hma_crossover_long:
                    long_conditions += 1
        
        # SHORT ENTRIES
        short_conditions = 0
        
        if htf_1d_bear:  # Only short when HTF is bearish
            if is_trend_regime:
                # Trend regime: HMA crossover + RSI confirmation
                if hma_crossover_short:
                    short_conditions += 2
                if hma_12h_bear and rsi_neutral:
                    short_conditions += 1
                if rsi_overbought:
                    short_conditions += 1
            elif is_range_regime:
                # Range regime: RSI mean reversion
                if rsi_extreme_overbought:
                    short_conditions += 2
                elif rsi_overbought:
                    short_conditions += 1
                if price_below_sma200:
                    short_conditions += 1
            else:
                # Transition zone: require multiple confirmations
                if hma_12h_bear and rsi_overbought:
                    short_conditions += 2
                if hma_crossover_short:
                    short_conditions += 1
        
        # Determine signal based on conditions
        if long_conditions >= 2 and short_conditions < 2:
            if long_conditions >= 3:
                desired_signal = SIZE_STRONG
                signal_strength = 2
            else:
                desired_signal = SIZE_BASE
                signal_strength = 1
        elif short_conditions >= 2 and long_conditions < 2:
            if short_conditions >= 3:
                desired_signal = -SIZE_STRONG
                signal_strength = 2
            else:
                desired_signal = -SIZE_BASE
                signal_strength = 1
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
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
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals