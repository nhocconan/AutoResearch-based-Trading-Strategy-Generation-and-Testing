#!/usr/bin/env python3
"""
Experiment #1333: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: Daily timeframe with weekly trend filter reduces whipsaw while maintaining
trade frequency. Connors RSI (CRSI) has proven 75% win rate for mean reversion entries.
Choppiness Index detects regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend (follow).
Combined with 1w HMA for macro bias, this should work in both bull and bear markets.

Key design:
1. 1w HMA(21) for macro trend filter (align with mtf_data)
2. 1d Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
3. Choppiness Index(14) for regime detection
4. Dual regime: mean revert in chop, trend follow otherwise
5. ATR(14) trailing stop at 2.5x for risk management
6. Size: 0.28 discrete levels

Target: 20-50 trades/year on 1d, Sharpe > 0.612, trades >= 40 train, >= 5 test
Timeframe: 1d
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
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=3):
    """Relative Strength Index - short period for Connors RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak component of Connors RSI
    Measures consecutive up/down days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    streak_rsi = np.full(n, np.nan)
    
    for i in range(period, n):
        # Count consecutive up days
        up_streak = 0
        for j in range(i, max(i - 20, 0), -1):
            if j == 0:
                break
            if close[j] > close[j - 1]:
                up_streak += 1
            else:
                break
        
        # Count consecutive down days
        down_streak = 0
        for j in range(i, max(i - 20, 0), -1):
            if j == 0:
                break
            if close[j] < close[j - 1]:
                down_streak += 1
            else:
                break
        
        # Streak value (positive for up, negative for down)
        if up_streak > 0:
            streak_value = up_streak
        elif down_streak > 0:
            streak_value = -down_streak
        else:
            streak_value = 0
        
        # Calculate RSI of streak over last 'period' days
        streak_values = []
        for k in range(i - period + 1, i + 1):
            if k < period:
                continue
            up_s = 0
            for j in range(k, max(k - 20, 0), -1):
                if j == 0:
                    break
                if close[j] > close[j - 1]:
                    up_s += 1
                else:
                    break
            down_s = 0
            for j in range(k, max(k - 20, 0), -1):
                if j == 0:
                    break
                if close[j] < close[j - 1]:
                    down_s += 1
                else:
                    break
            if up_s > 0:
                streak_values.append(up_s)
            elif down_s > 0:
                streak_values.append(-down_s)
            else:
                streak_values.append(0)
        
        if len(streak_values) >= period:
            # Convert streak to RSI-like scale (0-100)
            streak_arr = np.array(streak_values[-period:])
            gain_streak = np.where(streak_arr > 0, streak_arr, 0)
            loss_streak = np.where(streak_arr < 0, -streak_arr, 0)
            
            avg_gain = np.mean(gain_streak)
            avg_loss = np.mean(loss_streak)
            
            if avg_loss > 1e-10:
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            else:
                streak_rsi[i] = 100.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank component of Connors RSI
    Measures current price change vs past 'period' days
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    pct_rank = np.full(n, np.nan)
    
    for i in range(period, n):
        # Current 1-day return
        current_return = (close[i] - close[i - 1]) / close[i - 1] if close[i - 1] > 1e-10 else 0
        
        # Count how many of past 'period' returns are less than current
        count_below = 0
        total = 0
        for j in range(i - period + 1, i):
            if j == 0:
                continue
            past_return = (close[j] - close[j - 1]) / close[j - 1] if close[j - 1] > 1e-10 else 0
            total += 1
            if past_return < current_return:
                count_below += 1
        
        if total > 0:
            pct_rank[i] = 100.0 * count_below / total
    
    return pct_rank

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries
    """
    rsi = calculate_rsi(close, rsi_period)
    streak_rsi = calculate_rsi_streak(close, streak_period)
    pct_rank = calculate_percent_rank(close, pr_period)
    
    n = len(close)
    crsi = np.full(n, np.nan)
    
    for i in range(pr_period, n):
        if not np.isnan(rsi[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - detects ranging vs trending markets
    CHOP > 61.8 = range (mean revert)
    CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Highest high and lowest low over period
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        # Sum of ATR over period (using simple true range)
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
            atr_sum += tr
        
        if atr_sum > 1e-10 and (highest_high - lowest_low) > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    return sma

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
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1w HMA) ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        is_chop = chop[i] > 61.8  # Range market
        is_trend = chop[i] < 38.2  # Trending market
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === MEAN REVERSION IN CHOPPY MARKET ===
        if is_chop:
            # Long: CRSI extremely oversold (< 10) + above SMA200 support
            if crsi[i] < 10.0 and above_sma200:
                desired_signal = BASE_SIZE
            # Short: CRSI extremely overbought (> 90) + below SMA200 resistance
            elif crsi[i] > 90.0 and below_sma200:
                desired_signal = -BASE_SIZE
            # Moderate mean reversion (wider bands for more trades)
            elif crsi[i] < 20.0 and macro_bull:
                desired_signal = BASE_SIZE * 0.5
            elif crsi[i] > 80.0 and macro_bear:
                desired_signal = -BASE_SIZE * 0.5
        
        # === TREND FOLLOWING IN TRENDING MARKET ===
        elif is_trend:
            # Long: Macro bull + CRSI pullback (30-50) + above SMA200
            if macro_bull and 30.0 <= crsi[i] <= 50.0 and above_sma200:
                desired_signal = BASE_SIZE
            # Short: Macro bear + CRSI bounce (50-70) + below SMA200
            elif macro_bear and 50.0 <= crsi[i] <= 70.0 and below_sma200:
                desired_signal = -BASE_SIZE
            # Strong trend continuation
            elif macro_bull and crsi[i] > 50.0 and above_sma200:
                desired_signal = BASE_SIZE * 0.5
            elif macro_bear and crsi[i] < 50.0 and below_sma200:
                desired_signal = -BASE_SIZE * 0.5
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        else:
            # Use macro trend with CRSI extremes
            if macro_bull and crsi[i] < 25.0:
                desired_signal = BASE_SIZE * 0.5
            elif macro_bear and crsi[i] > 75.0:
                desired_signal = -BASE_SIZE * 0.5
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.14:
            final_signal = BASE_SIZE
        elif desired_signal > 0.07:
            final_signal = BASE_SIZE * 0.5
        elif desired_signal < -0.14:
            final_signal = -BASE_SIZE
        elif desired_signal < -0.07:
            final_signal = -BASE_SIZE * 0.5
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