#!/usr/bin/env python3
"""
Experiment #1554: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Daily timeframe with weekly trend filter provides optimal balance of
trade frequency (20-50/year) and signal quality. Connors RSI (CRSI) has documented
75% win rate for mean reversion entries. Combined with Choppiness Index regime
detection and 1w HMA trend bias, this should work in bear/range markets (2025+).

Key components:
1. 1w HMA(21) for major trend bias (avoid counter-trend in strong trends)
2. 1d Connors RSI(3,2,100) for entry timing — proven mean reversion signal
3. 1d Choppiness Index(14) for regime detection (chop>61.8 = mean revert, chop<38.2 = trend)
4. 1d SMA(200) for long-term trend filter
5. ATR(14) trailing stoploss (2.5x ATR)
6. Discrete sizing: 0.0, ±0.25, ±0.30 (minimize fee churn)

Connors RSI formula:
- RSI(3): 3-period RSI on close
- RSI_Streak(2): 2-period RSI on streak length (consecutive up/down days)
- PercentRank(100): percentile rank of 1-day price change over 100 periods
- CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3/test):
- LONG: 1w_HMA bullish + CRSI<15 + price>SMA200 + (CHOP>50 OR CHOP<40)
- SHORT: 1w_HMA bearish + CRSI>85 + price<SMA200 + (CHOP>50 OR CHOP<40)

Why this should work:
- Connors RSI is proven mean reversion signal (75% win rate in literature)
- 1d TF = natural 25-40 trades/year (fee-efficient)
- 1w HMA filter prevents major counter-trend disasters
- Choppiness filter adapts to regime without over-complicating
- Loose CRSI thresholds (15/85, not 10/90) guarantee trade generation

Target: Sharpe>0.6, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v2"
timeframe = "1d"
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
    Streak RSI for Connors RSI
    Measures RSI of consecutive up/down day streaks
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak length
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on absolute streak values
    streak_abs = np.abs(streak)
    streak_abs = np.where(streak_abs == 0, 0.01, streak_abs)  # avoid zero
    
    # Use simple RSI calculation on streak
    delta = np.diff(streak_abs)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi_streak[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """
    Percent Rank for Connors RSI
    Percentile rank of 1-day price change over lookback period
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate 1-day returns
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i-1] > 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        window = returns[i - period + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            current = returns[i]
            rank = np.sum(valid < current) / len(valid) * 100
            percent_rank[i] = rank
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < pr_period + 5:
        return np.full(n, np.nan)
    
    rsi_3 = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_streak_rsi(close, streak_period)
    percent_rank = calculate_percent_rank(close, pr_period)
    
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(pr_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
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

def calculate_sma(close, period=200):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(sma_200[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_range_regime = chop > 50.0  # Lower threshold to allow more trades
        is_trend_regime = chop < 40.0
        
        # === TREND DIRECTION (1w HMA bias) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === LONG TERM TREND (SMA200) ===
        price_above_sma = close[i] > sma_200[i]
        price_below_sma = close[i] < sma_200[i]
        
        # === CONNORS RSI ===
        crsi_val = crsi[i]
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # LONG: 1w bullish + CRSI oversold + price above SMA200
        if price_above_1w and crsi_val < 20 and price_above_sma:
            desired_signal = SIZE_STRONG if is_range_regime else SIZE_BASE
        
        # SHORT: 1w bearish + CRSI overbought + price below SMA200
        elif price_below_1w and crsi_val > 80 and price_below_sma:
            desired_signal = -SIZE_STRONG if is_range_regime else -SIZE_BASE
        
        # Additional mean reversion entries in range regime (looser conditions)
        if is_range_regime and desired_signal == 0.0:
            # Long on extreme CRSI even without 1w confirmation
            if crsi_val < 12 and price_above_sma:
                desired_signal = SIZE_BASE
            # Short on extreme CRSI even without 1w confirmation
            elif crsi_val > 88 and price_below_sma:
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