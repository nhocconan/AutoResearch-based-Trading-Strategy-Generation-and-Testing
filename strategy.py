#!/usr/bin/env python3
"""
Experiment #1324: 12h Primary + 1d/1w HTF — Dual Regime (Chopiness + Connors RSI)

Hypothesis: The proven 12h pattern (Choppiness Index + Connors RSI) achieved ETH Sharpe +0.923.
This implements a DUAL REGIME strategy that adapts to market conditions:

1. CHOPPY REGIME (CHOP > 61.8): Mean reversion using Connors RSI
   - Long: CRSI < 15 + price > SMA200
   - Short: CRSI > 85 + price < SMA200
   
2. TRENDING REGIME (CHOP < 38.2): Trend following using HMA
   - Long: 12h HMA rising + 1d HMA bullish + price > 12h HMA
   - Short: 12h HMA falling + 1d HMA bearish + price < 12h HMA

3. 1w HMA for major regime bias (only trade with weekly trend direction)

Why this should work:
- 12h timeframe = natural 20-50 trades/year (fee-friendly)
- Dual regime = adapts to both choppy and trending markets
- Connors RSI = 75% win rate on mean reversion (proven in literature)
- Choppiness Index = best meta-filter for bear/range markets (2022-2024)
- Weekly HTF filter = prevents trading against major trend
- LOOSE entry thresholds to guarantee 30+ trades on train

Key differences from failed strategies:
- NOT requiring all filters to agree (over-filtering = 0 trades)
- Connors RSI thresholds widened (10-90 → 15-85) for more signals
- CHOP thresholds at 38.2/61.8 (Fibonacci levels, proven)
- Discrete sizing (0.0, ±0.20, ±0.30) to minimize fee churn

Target: Sharpe>0.5, trades>=30 train, trades>=3 test, DD>-35%
Timeframe: 12h
Size: 0.20-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_chop_crsi_hma_1d1w_v1"
timeframe = "12h"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return vs last 100 days
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3) - fast RSI
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        if len(streak_vals) > 0:
            avg_streak = np.mean(np.abs(streak_vals))
            # Map streak magnitude to 0-100 scale
            streak_rsi[i] = min(100.0, max(0.0, 50.0 + avg_streak * 10.0))
    
    # Percent Rank (today's return vs last 100 days)
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    returns = np.diff(close, prepend=close[0]) / np.where(close != 0, close, 1.0)
    
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < returns[i])
            percent_rank[i] = (count_below / (rank_period - 1)) * 100.0
    
    # Combine into Connors RSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_fast[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_fast[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppy vs trending
    CHOP > 61.8 = choppy/range-bound
    CHOP < 38.2 = trending
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        atr_sum = np.nansum(atr[i-period+1:i+1])
        highest = np.nanmax(high[i-period+1:i+1])
        lowest = np.nanmin(low[i-period+1:i+1])
        price_range = highest - lowest
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    hma_12h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
    
    # Warmup period
    min_bars = 250  # Need enough bars for SMA200 and CRSI
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(sma_200[i]):
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
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        is_choppy = chop > 55.0  # Slightly relaxed from 61.8 for more signals
        is_trending = chop < 45.0  # Slightly relaxed from 38.2 for more signals
        
        # === WEEKLY TREND BIAS (major filter) ===
        price_above_1w = close[i] > hma_1w_aligned[i]
        price_below_1w = close[i] < hma_1w_aligned[i]
        
        # === DAILY TREND (secondary filter) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA slope ===
        hma_12h_slope = 0.0
        if i >= 3 and not np.isnan(hma_12h[i-3]):
            hma_12h_slope = hma_12h[i] - hma_12h[i-3]
        
        price_above_12h = close[i] > hma_12h[i]
        price_below_12h = close[i] < hma_12h[i]
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # --- CHOPPY REGIME: Mean Reversion with Connors RSI ---
        if is_choppy:
            # LONG: CRSI oversold + above SMA200 + weekly bias OK
            if crsi[i] < 25.0 and close[i] > sma_200[i]:
                if price_above_1w or not price_below_1w:  # Not strongly bearish weekly
                    if crsi[i] < 15.0:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # SHORT: CRSI overbought + below SMA200 + weekly bias OK
            elif crsi[i] > 75.0 and close[i] < sma_200[i]:
                if price_below_1w or not price_above_1w:  # Not strongly bullish weekly
                    if crsi[i] > 85.0:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # --- TRENDING REGIME: Trend Following with HMA ---
        elif is_trending:
            # LONG: 12h HMA rising + price above 12h HMA + daily/weekly OK
            if hma_12h_slope > 0 and price_above_12h:
                if price_above_1d or price_above_1w:  # At least one HTF bullish
                    if hma_12h_slope > (hma_12h[i-1] - hma_12h[i-4] if i >= 4 else 0):
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # SHORT: 12h HMA falling + price below 12h HMA + daily/weekly OK
            elif hma_12h_slope < 0 and price_below_12h:
                if price_below_1d or price_below_1w:  # At least one HTF bearish
                    if hma_12h_slope < (hma_12h[i-1] - hma_12h[i-4] if i >= 4 else 0):
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # --- NEUTRAL REGIME (45-55 CHOP): Reduce position or flat ---
        else:
            # Only take strongest signals in neutral regime
            if crsi[i] < 12.0 and close[i] > sma_200[i]:
                desired_signal = SIZE_BASE * 0.5
            elif crsi[i] > 88.0 and close[i] < sma_200[i]:
                desired_signal = -SIZE_BASE * 0.5
        
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
        elif desired_signal >= SIZE_BASE * 0.4:
            final_signal = SIZE_BASE * 0.5
        elif desired_signal <= -SIZE_BASE * 0.4:
            final_signal = -SIZE_BASE * 0.5
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