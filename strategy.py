#!/usr/bin/env python3
"""
Experiment #1467: 6h Primary + 1d HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: 6h timeframe is underexplored (ZERO prior experiments). Connors RSI (CRSI)
has documented 75% win rate for mean-reversion in crypto. Combined with 1d HMA trend
bias and Choppiness Index regime filter, this should generate consistent trades with
positive Sharpe even in bear/range markets (2025 test period).

Key components:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Proven mean-reversion indicator with sharp entry signals
2. 1d HMA(21) for major trend bias (only long above, only short below)
3. Choppiness Index(14) regime filter:
   - CHOP > 55 = favor mean-reversion (CRSI extremes)
   - CHOP < 45 = favor trend continuation (pullback entries)
4. ATR(14) volatility filter: skip entries if ATR ratio > 2.5 (extreme vol)
5. Volume confirmation: entry volume > 0.8 * 20-bar avg volume
6. ATR(14) trailing stoploss (2.5x ATR)
7. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Why this should work on 6h:
- CRSI generates frequent signals (RSI(3) is very responsive)
- 6h = ~4 bars/day, natural 40-60 trades/year target
- 1d HMA prevents major counter-trend positions
- CHOP filter adapts to regime (mean-revert in chop, trend in breakout)
- LOOSE CRSI thresholds (15/85 not 10/90) guarantee trade generation

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1d_HMA bullish + CRSI<20 + (CHOP>55 OR pullback to HMA16) + vol_confirm
- SHORT: 1d_HMA bearish + CRSI>80 + (CHOP>55 OR rally to HMA16) + vol_confirm

Target: Sharpe>0.6, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_crsi_chop_regime_hma_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_streak_rsi(close, period=2):
    """
    Connors RSI Streak component: RSI of consecutive up/down streak length
    Measures how many consecutive bars price has moved in same direction
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak lengths
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert to absolute streak length for RSI calculation
    abs_streak = np.abs(streak)
    
    # Calculate RSI on streak lengths (inverted: long streak = overbought)
    delta = np.diff(abs_streak)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    streak_rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    # Invert: long up-streak = high RSI = overbought (bearish for mean reversion)
    return 100 - streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank: percentage of values in lookback period below current value
    Connors RSI component - measures relative position in recent range
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < window[-1])
            pr[i] = 100.0 * count_below / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100, extremes <10 or >90 signal mean-reversion opportunities
    """
    rsi_close = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_streak_rsi(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(len(close), np.nan, dtype=np.float64)
    mask = ~np.isnan(rsi_close) & ~np.isnan(rsi_streak) & ~np.isnan(pr)
    crsi[mask] = (rsi_close[mask] + rsi_streak[mask] + pr[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending
    CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup period
    min_bars = 150
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Volatility filter - skip extreme volatility
        if not np.isnan(atr_30[i]) and atr_30[i] > 1e-10:
            atr_ratio = atr_14[i] / atr_30[i]
            if atr_ratio > 2.5:
                signals[i] = 0.0
                if in_position:
                    in_position = False
                    position_side = 0
                continue
        
        # Volume confirmation
        vol_confirm = True
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 0.8 * vol_sma[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_range_regime = chop > 55.0
        is_trend_regime = chop < 45.0
        
        # === TREND DIRECTION (1d HMA bias) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA CROSSOVER (momentum) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === Connors RSI ===
        crsi_val = crsi[i]
        crsi_oversold = crsi_val < 20.0
        crsi_overbought = crsi_val > 80.0
        
        # === Pullback to HMA16 (for trend regime) ===
        pullback_long = hma_bullish and close[i] < hma_16[i] * 1.002 and close[i] > hma_16[i] * 0.995
        pullback_short = hma_bearish and close[i] > hma_16[i] * 0.998 and close[i] < hma_16[i] * 1.005
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # RANGE REGIME: CRSI mean reversion (primary signal)
        if is_range_regime:
            # LONG: CRSI oversold + 1d bullish bias + volume confirm
            if crsi_oversold and price_above_1d and vol_confirm:
                desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + 1d bearish bias + volume confirm
            elif crsi_overbought and price_below_1d and vol_confirm:
                desired_signal = -SIZE_BASE
        
        # TREND REGIME: Pullback entries with CRSI confirmation
        elif is_trend_regime:
            # LONG: 1d bullish + pullback to HMA16 + CRSI not overbought
            if price_above_1d and pullback_long and crsi_val < 60 and vol_confirm:
                desired_signal = SIZE_STRONG
            
            # SHORT: 1d bearish + rally to HMA16 + CRSI not oversold
            elif price_below_1d and pullback_short and crsi_val > 40 and vol_confirm:
                desired_signal = -SIZE_STRONG
        
        # NEUTRAL REGIME: Use CRSI extremes with 1d confirmation
        else:
            # LONG: 1d bullish + CRSI very oversold
            if price_above_1d and crsi_val < 15 and vol_confirm:
                desired_signal = SIZE_BASE
            
            # SHORT: 1d bearish + CRSI very overbought
            elif price_below_1d and crsi_val > 85 and vol_confirm:
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