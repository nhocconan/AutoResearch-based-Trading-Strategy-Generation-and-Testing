#!/usr/bin/env python3
"""
Experiment #1028: 4h Primary + 12h/1d HTF — Dual Regime HMA + RSI + Choppiness

Hypothesis: 4h timeframe captures multi-day swings with optimal trade frequency (20-50/year).
Using Choppiness Index to switch between mean-reversion (range) and trend-following (trend)
regimes, combined with 12h/1d HMA for bias and RSI for entry timing.

Key innovations:
1. Choppiness Index (CHOP 14): >55 = range (mean revert), <45 = trend (trend follow)
2. Dual HMA filter: 12h HMA(21) for intermediate trend, 1d HMA(21) for long-term bias
3. Regime-adaptive entries with LOOSE conditions to guarantee trades:
   - Range: CRSI<25 long, CRSI>75 short (wider than typical 15/85)
   - Trend: Price vs HMA alignment + RSI 40-60 filter (not extreme)
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn

Why this should work:
- 4h captures meaningful moves without noise (proven in exp #1002)
- Choppiness filter avoids whipsaws in 2022-2023 range markets
- LOOSE entry conditions guarantee 30+ trades (fixes #1 failure mode)
- HTF bias (12h/1d) ensures we trade with higher timeframe flow
- Discrete sizing reduces fee churn from signal changes

Entry conditions (LOOSE to guarantee trades):
- LONG range: CHOP>50 + CRSI<30 + price>1d_HMA*0.97
- LONG trend: CHOP<50 + price>12h_HMA>1d_HMA + RSI(14)>40
- SHORT range: CHOP>50 + CRSI>70 + price<1d_HMA*1.03
- SHORT trend: CHOP<50 + price<12h_HMA<1d_HMA + RSI(14)<60

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_hma_rsi_dual_regime_12h1d_v1"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
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
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range > 1e-10 and atr_sum > 0:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - combines 3 components for mean reversion signals
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    rsi_3[:rsi_period] = np.nan
    
    # Component 2: RSI of Streak
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_delta = np.diff(streak, prepend=streak[0])
    streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_rs = np.divide(avg_streak_gain, avg_streak_loss, out=np.zeros_like(avg_streak_gain), where=avg_streak_loss != 0)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi[:streak_period] = np.nan
    
    # Component 3: Percent Rank
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window < close[i])
            percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine components
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
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
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 50.0  # Range market (looser threshold)
        is_trending = chop_14[i] < 50.0  # Trend market
        
        # === HTF BIAS (Dual HMA alignment) ===
        price_vs_12h_hma = close[i] / hma_12h_aligned[i]
        price_vs_1d_hma = close[i] / hma_1d_aligned[i]
        hma_12h_vs_1d = hma_12h_aligned[i] / hma_1d_aligned[i]
        
        hma_12h_bull = price_vs_12h_hma > 1.0
        hma_12h_bear = price_vs_12h_hma < 1.0
        hma_1d_bull = price_vs_1d_hma > 1.0
        hma_1d_bear = price_vs_1d_hma < 1.0
        
        # Strong trend alignment
        strong_bull = hma_12h_bull and hma_1d_bull and hma_12h_vs_1d > 1.0
        strong_bear = hma_12h_bear and hma_1d_bear and hma_12h_vs_1d < 1.0
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE, LOOSE CONDITIONS) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION MODE - use Connors RSI extremes (LOOSE)
            # Long when CRSI oversold + price near/near 1d HMA
            if crsi[i] < 30.0 and price_vs_1d_hma > 0.97:
                desired_signal = SIZE_BASE
            # Short when CRSI overbought + price near/above 1d HMA
            elif crsi[i] > 70.0 and price_vs_1d_hma < 1.03:
                desired_signal = -SIZE_BASE
            # Stronger signals at more extreme CRSI
            elif crsi[i] < 20.0 and price_vs_1d_hma > 0.95:
                desired_signal = SIZE_STRONG
            elif crsi[i] > 80.0 and price_vs_1d_hma < 1.05:
                desired_signal = -SIZE_STRONG
        
        else:
            # TREND FOLLOWING MODE - use HMA alignment + RSI filter (LOOSE)
            # Long in uptrend with RSI not overbought
            if strong_bull and rsi_14[i] > 40.0 and rsi_14[i] < 75.0:
                desired_signal = SIZE_STRONG
            # Short in downtrend with RSI not oversold
            elif strong_bear and rsi_14[i] < 60.0 and rsi_14[i] > 25.0:
                desired_signal = -SIZE_STRONG
            # Weaker trend signals (single HMA alignment)
            elif hma_12h_bull and hma_1d_bull and rsi_14[i] > 45.0:
                desired_signal = SIZE_BASE
            elif hma_12h_bear and hma_1d_bear and rsi_14[i] < 55.0:
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
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = close[i] - 2.5 * entry_atr
                else:
                    stop_price = close[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals