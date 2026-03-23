#!/usr/bin/env python3
"""
Experiment #1243: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: Research shows Connors RSI (CRSI) achieves 75% win rate for mean reversion
entries, and Choppiness Index is the best meta-filter for detecting trend vs range regimes.
Key innovations:
1. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven 75% win rate
2. Choppiness Index regime: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trend
3. 1w HMA for macro trend bias (only long when price > 1w HMA in bull, vice versa)
4. Dual regime logic: trend-follow in trending markets, mean-revert in choppy
5. Asymmetric sizing: 0.30 for trend trades, 0.25 for mean-revert trades
6. ATR 2.5x trailing stoploss on all positions

Target: Sharpe > 0.612, trades >= 20/year (80+ train, 12+ test)
Timeframe: 1d (20-50 trades/year target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (2) - measures consecutive up/down days
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    abs_streak = np.abs(streak)
    streak_gain = np.where(streak > 0, abs_streak, 0)
    streak_loss = np.where(streak < 0, abs_streak, 0)
    
    streak_gain_smooth = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    streak_loss_smooth = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    mask = streak_loss_smooth > 1e-10
    streak_rs = np.zeros(n)
    streak_rs[mask] = streak_gain_smooth[mask] / (streak_loss_smooth[mask] + 1e-10)
    streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi[:streak_period] = np.nan
    
    # Percent Rank (100) - where current return ranks vs last 100 days
    percent_rank = np.full(n, np.nan)
    returns = np.zeros(n)
    returns[1:] = np.diff(close) / close[:-1] * 100
    
    for i in range(rank_period, n):
        window = returns[i - rank_period + 1:i + 1]
        current = returns[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine into CRSI
    valid_mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[valid_mask] = (rsi_short[valid_mask] + streak_rsi[valid_mask] + percent_rank[valid_mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - detects trend vs range regimes
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Daily HMA for trend confirmation
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    TREND_SIZE = 0.30
    MEAN_REVERT_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Signal persistence buffer
    signal_buffer = 0
    last_signal = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # Range market - mean revert
        is_trending = chop[i] < 38.2  # Trend market - trend follow
        
        # === DAILY TREND ===
        daily_bull = hma_21[i] > hma_50[i]
        daily_bear = hma_21[i] < hma_50[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Strong buy signal for mean reversion
        crsi_overbought = crsi[i] > 85  # Strong sell signal for mean reversion
        crsi_pullback_long = crsi[i] < 40  # Moderate pullback in uptrend
        crsi_pullback_short = crsi[i] > 60  # Moderate rally in downtrend
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow macro trend with pullback entries
        if is_trending:
            # Long: Macro bull + Daily bull + CRSI pullback
            if macro_bull and daily_bull and crsi_pullback_long:
                desired_signal = TREND_SIZE
            # Short: Macro bear + Daily bear + CRSI pullback
            elif macro_bear and daily_bear and crsi_pullback_short:
                desired_signal = -TREND_SIZE
        
        # CHOPPY REGIME: Mean revert at extremes (ignore macro trend)
        elif is_choppy:
            # Long: CRSI deeply oversold
            if crsi_oversold:
                desired_signal = MEAN_REVERT_SIZE
            # Short: CRSI deeply overbought
            elif crsi_overbought:
                desired_signal = -MEAN_REVERT_SIZE
        
        # NEUTRAL REGIME: Use macro trend bias with stricter CRSI
        else:
            if macro_bull and crsi[i] < 35:
                desired_signal = MEAN_REVERT_SIZE
            elif macro_bear and crsi[i] > 65:
                desired_signal = -MEAN_REVERT_SIZE
        
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
        
        # === SIGNAL PERSISTENCE (prevent noise flipping) ===
        if desired_signal != last_signal:
            signal_buffer += 1
            if signal_buffer >= 2:
                last_signal = desired_signal
                signal_buffer = 0
        else:
            signal_buffer = 0
        
        final_signal = last_signal
        
        # === DISCRETIZE SIGNAL VALUES ===
        if final_signal > 0:
            final_signal = TREND_SIZE if is_trending else MEAN_REVERT_SIZE
        elif final_signal < 0:
            final_signal = -TREND_SIZE if is_trending else -MEAN_REVERT_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals