#!/usr/bin/env python3
"""
Experiment #1288: 4h Primary + 12h/1d HTF — Dual Regime (Choppiness + Connors RSI + HMA)

Hypothesis: The dual-regime approach adapts to market conditions automatically.
2021 was trending (trend-follow works), 2022 was crash (short trend works),
2023-2024 was choppy (mean-reversion works), 2025 is bear/range (mean-reversion + short trend).

Key innovations vs failed strategies:
1. CHOPPINESS INDEX (CHOP) regime detection: CHOP>61.8=range, CHOP<38.2=trend
2. CONNORS RSI for mean-reversion entries in choppy regime (75% win rate in literature)
3. HMA trend + ROC momentum for trending regime
4. 12h HMA for intermediate trend, 1d HMA for major bias
5. ATR(14) 2.5x trailing stop for all positions
6. LOOSE entry thresholds to guarantee 20-50 trades/year on 4h

Why this should beat Sharpe=0.447:
- Adapts to ALL market regimes (bull/bear/range)
- Connors RSI mean-reversion works in 2023-2024 chop and 2025 bear
- HMA trend works in 2021 bull and 2022 crash
- 4h timeframe = natural 20-50 trades/year (fee-friendly)
- Discrete sizing (0.0, ±0.25, ±0.30) = minimal fee churn

Entry logic:
- CHOPPY (CHOP>55): Long when CRSI<15 + price>SMA200, Short when CRSI>85 + price<SMA200
- TRENDING (CHOP<45): Long when 12h_HMA rising + 1d_HMA bullish + ROC>5
                         Short when 12h_HMA falling + 1d_HMA bearish + ROC<-5
- TRANSITION (45<=CHOP<=55): No new entries, hold existing positions

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dual_regime_chop_crsi_hma_12h1d_v1"
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

def calculate_roc(close, period=10):
    """Rate of Change - momentum indicator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    roc = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if close[i - period] != 0:
            roc[i] = ((close[i] - close[i - period]) / close[i - period]) * 100.0
    
    return roc

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        elif avg_gain[i] > 0:
            rsi[i] = 100
        else:
            rsi[i] = 50
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppiness vs trending
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR for each bar (True Range, not smoothed)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 0:
            choppiness[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - composite mean-reversion indicator
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(3): Very short-term RSI for oversold/overbought
    RSI(Streak): RSI of consecutive up/down days
    PercentRank: Percentile rank of daily returns over 100 days
    
    CRSI < 10 = extreme oversold (long signal)
    CRSI > 90 = extreme overbought (short signal)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Calculate streak (consecutive up/down days)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI of streak (use absolute values for RSI calculation)
    # Convert streak to "gains/losses" for RSI
    streak_delta = np.diff(streak)
    streak_delta = np.insert(streak_delta, 0, 0)
    streak_gain = np.where(streak_delta > 0, streak_delta, 0)
    streak_loss = np.where(streak_delta < 0, -streak_delta, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.full(n, np.nan, dtype=np.float64)
    for i in range(streak_period, n):
        if avg_streak_loss[i] != 0:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            rsi_streak[i] = 100 - (100 / (1 + rs))
        elif avg_streak_gain[i] > 0:
            rsi_streak[i] = 100
        else:
            rsi_streak[i] = 50
    
    # Percent Rank of daily returns over 100 days
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    returns = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i-1] != 0:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            count_below = np.sum(valid_window[:-1] < returns[i])  # Exclude current
            percent_rank[i] = count_below / (len(valid_window) - 1) * 100 if len(valid_window) > 1 else 50
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + rsi_streak[i] + percent_rank[i]) / 3.0
    
    return crsi

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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    roc_10 = calculate_roc(close, period=10)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_4h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
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
    
    # Warmup period (need 200 bars for SMA200 + 100 for CRSI rank)
    min_bars = 250
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(choppiness[i]):
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
        
        if np.isnan(hma_4h[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = choppiness[i]
        
        # Determine regime
        is_choppy = chop > 55.0  # Range/mean-reversion regime
        is_trending = chop < 45.0  # Trend-following regime
        # 45-55 = transition zone (hold existing, no new entries)
        
        # === TREND DIRECTION (12h HMA slope + 1d HMA bias) ===
        hma_12h_slope = 0.0
        if i >= 3 and not np.isnan(hma_12h_aligned[i-3]):
            hma_12h_slope = hma_12h_aligned[i] - hma_12h_aligned[i-3]
        
        # 1d HMA bias
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # 4h price vs 4h HMA for local confirmation
        price_above_4h = close[i] > hma_4h[i]
        price_below_4h = close[i] < hma_4h[i]
        
        # Price vs SMA200 for major trend filter
        price_above_200 = close[i] > sma_200[i]
        price_below_200 = close[i] < sma_200[i]
        
        # === MOMENTUM (ROC) ===
        roc = roc_10[i]
        
        # === CONNORS RSI (for choppy regime) ===
        crsi_val = crsi[i]
        
        # === ENTRY LOGIC (Dual Regime) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION REGIME - use Connors RSI
            # Long: CRSI < 15 (oversold) + price above SMA200 (bullish bias)
            if crsi_val < 15.0 and price_above_200:
                if crsi_val < 10.0:
                    desired_signal = SIZE_STRONG  # Extreme oversold
                else:
                    desired_signal = SIZE_BASE  # Moderate oversold
            
            # Short: CRSI > 85 (overbought) + price below SMA200 (bearish bias)
            elif crsi_val > 85.0 and price_below_200:
                if crsi_val > 90.0:
                    desired_signal = -SIZE_STRONG  # Extreme overbought
                else:
                    desired_signal = -SIZE_BASE  # Moderate overbought
        
        elif is_trending:
            # TREND FOLLOWING REGIME - use HMA + ROC
            # Long: 12h HMA rising + 1d bullish + ROC positive
            if hma_12h_slope > 0 and price_above_1d and price_above_4h:
                if roc > 5.0:
                    if roc > 10.0:
                        desired_signal = SIZE_STRONG
                    else:
                        desired_signal = SIZE_BASE
            
            # Short: 12h HMA falling + 1d bearish + ROC negative
            elif hma_12h_slope < 0 and price_below_1d and price_below_4h:
                if roc < -5.0:
                    if roc < -10.0:
                        desired_signal = -SIZE_STRONG
                    else:
                        desired_signal = -SIZE_BASE
        
        # Transition zone (45-55 chop): only close positions, no new entries
        # desired_signal stays 0.0
        
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