#!/usr/bin/env python3
"""
Experiment #897: 15m Primary + 4h/12h HTF — Fisher Transform + HMA Trend + Regime Adaptive

Hypothesis: 15m timeframe with 4h trend bias and 12h regime filter can capture
intraday moves while avoiding fee drag. Ehlers Fisher Transform provides proven
reversal signals in bear/range markets (unlike EMA crossovers which fail on BTC/ETH).
Key insight from failures: 15m strategies generated 0 trades due to overly strict
confluence. This strategy uses LOOSE entry conditions with OR logic to ensure
≥10 trades/train, ≥3/test per symbol.

Innovations:
1. 4h HMA(21) for HTF trend bias — direction filter only, not hard requirement
2. 12h Choppiness Index for regime — adjust Fisher thresholds by regime
3. Ehlers Fisher Transform(9) — catches reversals better than RSI in bear markets
4. Session preference (00-12 UTC) but NOT hard filter — just size adjustment
5. ATR(14) 2.5x trailing stoploss
6. Discrete sizing: 0.0, ±0.20, ±0.25 (smaller for 15m frequency)

Entry logic (LOOSE — OR not AND):
- LONG: 4h HMA bull + Fisher < -1.0 OR Fisher cross above -1.5
- SHORT: 4h HMA bear + Fisher > +1.0 OR Fisher cross below +1.5
- Range regime (CHOP>50): widen Fisher thresholds to ±0.8
- Trend regime (CHOP<50): tighten Fisher thresholds to ±1.5

Target: Sharpe>0.45, trades>=30 train, trades>=5 test, DD>-40%
Timeframe: 15m
Size: 0.20-0.25 discrete (smaller than 6h/12h due to higher frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_fisher_hma_regime_4h12h_v1"
timeframe = "15m"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, span):
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1)
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
    Converts price to a Gaussian normal distribution for clearer reversal signals
    
    Fisher = 0.5 * ln((1 + X) / (1 - X))
    Where X = 0.66 * ((price - LL) / (HH - LL) - 0.5) + 0.67 * prev_X
    
    Long signal: Fisher crosses above -1.5 (oversold reversal)
    Short signal: Fisher crosses below +1.5 (overbought reversal)
    """
    n = len(high)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            fisher[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            fisher_prev[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
            continue
        
        X = 0.66 * ((high[i] - lowest) / price_range - 0.5) + 0.67 * (
            fisher_prev[i-1] if i > 0 and not np.isnan(fisher_prev[i-1]) else 0.0
        )
        
        # Clamp X to avoid division by zero
        X = np.clip(X, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1 + X) / (1 - X))
        fisher_prev[i] = fisher[i-1] if i > 0 and not np.isnan(fisher[i-1]) else 0.0
    
    return fisher, fisher_prev

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Using 50 as threshold for regime detection
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
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    chop_12h_raw = calculate_choppiness(
        df_12h['high'].values, 
        df_12h['low'].values, 
        df_12h['close'].values, 
        period=14
    )
    chop_12h_aligned = align_htf_to_ltf(prices, df_12h, chop_12h_raw)
    
    # Calculate 15m indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(chop_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === REGIME (12h Choppiness) ===
        chop_ranging = chop_12h_aligned[i] > 50.0
        chop_trending = chop_12h_aligned[i] <= 50.0
        
        # Adjust Fisher thresholds by regime
        if chop_ranging:
            fisher_long_threshold = -0.8
            fisher_short_threshold = 0.8
            fisher_cross_long = -1.2
            fisher_cross_short = 1.2
        else:
            fisher_long_threshold = -1.5
            fisher_short_threshold = 1.5
            fisher_cross_long = -1.8
            fisher_cross_short = 1.8
        
        # === FISHER SIGNALS (LOOSE — OR LOGIC FOR TRADES) ===
        fisher_oversold = fisher[i] < fisher_long_threshold
        fisher_overbought = fisher[i] > fisher_short_threshold
        
        # Fisher crossover signals
        fisher_cross_up = (fisher_prev[i] <= fisher_cross_long) and (fisher[i] > fisher_cross_long)
        fisher_cross_down = (fisher_prev[i] >= fisher_cross_short) and (fisher[i] < fisher_cross_short)
        
        # === SESSION FILTER (00-12 UTC preferred but not hard) ===
        hour_utc = (open_time[i] // 3600000) % 24
        is_preferred_session = 0 <= hour_utc <= 12
        
        # === ENTRY LOGIC (LOOSE — ensure trades) ===
        desired_signal = 0.0
        
        # Long entries (OR logic — any condition triggers)
        if htf_4h_bull or not htf_4h_bear:  # Neutral or bull bias
            if fisher_oversold or fisher_cross_up:
                if fisher_cross_up:
                    desired_signal = SIZE_STRONG if is_preferred_session else SIZE_BASE
                else:
                    desired_signal = SIZE_BASE
        
        # Short entries (OR logic — any condition triggers)
        if htf_4h_bear or not htf_4h_bull:  # Neutral or bear bias
            if fisher_overbought or fisher_cross_down:
                if fisher_cross_down:
                    desired_signal = -SIZE_STRONG if is_preferred_session else -SIZE_BASE
                else:
                    desired_signal = -SIZE_BASE
        
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