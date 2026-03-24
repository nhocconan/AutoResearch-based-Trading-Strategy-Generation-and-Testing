#!/usr/bin/env python3
"""
Experiment #864: 12h Primary + 1d/1w HTF — Fisher Transform + Vol Spike Reversion + Regime Adaptive

Hypothesis: 12h timeframe captures optimal swing moves while avoiding noise. 
Ehlers Fisher Transform excels at catching reversals in bear/range markets (2022 crash, 2025 bear).
Volatility spike reversion (ATR ratio > 2.0) captures panic bottoms with high win rate.
Choppiness Index regime switch enables adaptive logic: mean-revert in chop, trend-follow otherwise.
1d HMA provides HTF bias, 1w HMA provides meta-trend filter.

Key innovations:
1. Ehlers Fisher Transform (period=9) - catches reversals better than RSI in bear markets
2. Volatility spike detection: ATR(7)/ATR(30) > 1.8 indicates panic/extreme
3. Bollinger Band (20, 2.5) for mean-reversion entries at extremes
4. Choppiness Index(14) regime: >50 = range (mean revert), <50 = trend
5. 1d HMA(21) + 1w HMA(21) dual HTF bias for trend confirmation
6. ATR(14) 2.5x trailing stop with hysteresis

Entry conditions (LOOSE to ensure ≥10 trades/train, ≥3/test):
- RANGE REGIME (CHOP>50): LONG = Fisher<-1.5 + vol_spike OR price<BB_lower
- RANGE REGIME (CHOP>50): SHORT = Fisher>+1.5 + vol_spike OR price>BB_upper
- TREND REGIME (CHOP<50): LONG = 1d HMA bull + 1w HMA bull + Fisher crossover up
- TREND REGIME (CHOP<50): SHORT = 1d HMA bear + 1w HMA bear + Fisher crossover down

Target: Sharpe>0.45, trades>=10 train, trades>=3 test, DD>-40%
Timeframe: 12h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_volspike_chop_regime_1d1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    if sqrt_n < 1:
        sqrt_n = 1
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Typical price
    typical = (high + low + np.roll(high + low, 1)) / 4
    typical[0] = typical[1] if n > 1 else high[0]
    
    # Normalize to -1 to +1 range
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest[i] = np.max(high[i - period + 1:i + 1])
        lowest[i] = np.min(low[i - period + 1:i + 1])
    
    # Avoid division by zero
    range_val = highest - lowest
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    normalized = 2 * (typical - lowest) / range_val - 1
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        if abs(normalized[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1 + normalized[i]) / (1 - normalized[i]))
            if i > 0 and not np.isnan(fisher[i-1]):
                trigger[i] = fisher[i-1]
            else:
                trigger[i] = fisher[i]
    
    return fisher, trigger

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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Bollinger Bands with configurable standard deviation"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    
    return upper, lower, width

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 50 as practical threshold
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    fisher, fisher_trigger = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, period=20, std_mult=2.5)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volatility spike ratio
    vol_ratio = np.full(n, np.nan)
    for i in range(30, n):
        if atr_30[i] > 1e-10 and not np.isnan(atr_7[i]):
            vol_ratio[i] = atr_7[i] / atr_30[i]
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === VOLATILITY SPIKE ===
        vol_spike = vol_ratio[i] > 1.8 if not np.isnan(vol_ratio[i]) else False
        
        # === FISHER SIGNALS ===
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        
        # Fisher crossover (loose for more trades)
        fisher_cross_long = False
        fisher_cross_short = False
        if i > 0 and not np.isnan(fisher[i-1]) and not np.isnan(fisher_trigger[i-1]):
            fisher_cross_long = fisher[i-1] < -1.0 and fisher[i] > -1.5
            fisher_cross_short = fisher[i-1] > 1.0 and fisher[i] < 1.5
        
        # === BOLLINGER BAND ===
        price_below_bb = close[i] < bb_lower[i]
        price_above_bb = close[i] > bb_upper[i]
        
        # === CHOPPINESS REGIME ===
        chop_ranging = chop_14[i] >= 50.0
        chop_trending = chop_14[i] < 50.0
        
        # === ENTRY LOGIC (REGIME ADAPTIVE + LOOSE) ===
        desired_signal = 0.0
        
        if chop_ranging:
            # RANGE REGIME: Mean reversion at extremes
            # LONG: Fisher oversold OR price below BB + vol spike
            if fisher_oversold or (price_below_bb and vol_spike):
                if htf_1d_bull or htf_1w_bull:
                    # HTF bias confirms
                    desired_signal = SIZE_STRONG if vol_spike else SIZE_BASE
                else:
                    # No HTF bias but extreme conditions
                    desired_signal = SIZE_BASE * 0.8
            
            # SHORT: Fisher overbought OR price above BB + vol spike
            elif fisher_overbought or (price_above_bb and vol_spike):
                if htf_1d_bear or htf_1w_bear:
                    desired_signal = -SIZE_STRONG if vol_spike else -SIZE_BASE
                else:
                    desired_signal = -SIZE_BASE * 0.8
        
        else:
            # TREND REGIME: Follow HTF direction
            if htf_1d_bull and htf_1w_bull:
                # Strong bullish trend
                if fisher_cross_long or (fisher[i] > -1.0 and not fisher_overbought):
                    desired_signal = SIZE_BASE
                if vol_spike and fisher_oversold:
                    # Vol spike pullback in uptrend = strong buy
                    desired_signal = SIZE_STRONG
            
            elif htf_1d_bear and htf_1w_bear:
                # Strong bearish trend
                if fisher_cross_short or (fisher[i] < 1.0 and not fisher_oversold):
                    desired_signal = -SIZE_BASE
                if vol_spike and fisher_overbought:
                    # Vol spike rally in downtrend = strong sell
                    desired_signal = -SIZE_STRONG
        
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
        if desired_signal >= SIZE_STRONG * 0.8:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.8:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.8:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.8:
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