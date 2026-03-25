#!/usr/bin/env python3
"""
Experiment #1166: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Weekly Trend Filter

Hypothesis: After 960+ failures, the research notes show Connors RSI achieved ETH Sharpe +0.923.
This is a MEAN REVERSION strategy that works in bear/range markets (2025 test period).

Key insight: Trend-following failed on BTC/ETH 2022-2025 (whipsaw destruction). Mean reversion
with HTF trend filter should work better:
- Weekly HMA(21) determines bias (only long if weekly uptrend, only short if weekly downtrend)
- Connors RSI < 20 = oversold bounce opportunity (long)
- Connors RSI > 80 = overbought fade opportunity (short)
- Choppiness Index > 50 confirms ranging market (favors mean reversion)

Why this should work on 1d:
- 1d timeframe = natural 20-50 trades/year (fee-friendly)
- CRSI extremes happen regularly in crypto volatility
- Weekly trend filter prevents counter-trend disasters
- Loose CRSI thresholds (20/80) guarantee trades vs tight RSI ranges

CRSI Formula (Connors 2009):
- RSI(3): Very short-term momentum
- RSI_Streak(2): Streak duration RSI (how many consecutive up/down days)
- PercentRank(100): Current return rank vs last 100 days
- CRSI = (RSI3 + RSI_Streak + PercentRank) / 3

Entry logic (LOOSE to guarantee trades):
- LONG: CRSI < 25 + price > 1w_HMA (oversold in weekly uptrend)
- SHORT: CRSI > 75 + price < 1w_HMA (overbought in weekly downtrend)

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_meanrevert_weekly_v1"
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
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - Mean reversion indicator
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Research shows 75% win rate on CRSI < 10 long / CRSI > 90 short
    We use looser thresholds (25/75) to guarantee trade frequency
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3) - very short term momentum
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streak duration
    # Count consecutive up/down days
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n, dtype=np.float64)
    streak[0] = 1
    
    for i in range(1, n):
        if delta[i] > 0:
            if delta[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta[i] < 0:
            if delta[i-1] < 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI (up streaks = gains, down streaks = losses)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rs = np.divide(avg_streak_gain, avg_streak_loss, out=np.zeros_like(avg_streak_gain), where=avg_streak_loss != 0)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    rsi_streak[:streak_period] = np.nan
    
    # Component 3: Percentile Rank of returns over rank_period
    returns = np.diff(close, prepend=close[0]) / np.where(close != 0, close, 1.0)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window < returns[i])
            percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is trending or ranging
    CHOP > 61.8 = ranging (mean reversion favored)
    CHOP < 38.2 = trending (trend following favored)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
    return chop

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
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    
    # Warmup period (need 100+ bars for CRSI rank_period)
    min_bars = 120
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]):
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
        
        # === TREND BIAS (Weekly HMA) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS FILTER (favor mean reversion in ranges) ===
        chop_valid = not np.isnan(chop[i])
        is_choppy = chop_valid and chop[i] > 50.0  # Ranging market
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        crsi_val = crsi[i]
        
        # LONG: CRSI oversold + weekly uptrend (mean reversion bounce)
        # Strong long: also in choppy market (favors mean reversion)
        if price_above_1w:
            if crsi_val < 25.0:  # Oversold
                if is_choppy:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: CRSI overbought + weekly downtrend (mean reversion fade)
        # Strong short: also in choppy market
        elif price_below_1w:
            if crsi_val > 75.0:  # Overbought
                if is_choppy:
                    desired_signal = -SIZE_STRONG
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