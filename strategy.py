#!/usr/bin/env python3
"""
Experiment #1251: 4h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime

Hypothesis: #1239 HMA+RSI got Sharpe=0.077 (barely positive). Research shows Connors RSI
(CRSI) achieves 75% win rate in mean-reversion regimes. Key changes:
1. Connors RSI instead of standard RSI (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
2. Choppiness Index regime filter (CHOP>61.8 = range, CHOP<38.2 = trend)
3. 1w HMA as additional macro filter (triple HTF confirmation)
4. Remove hysteresis buffer (was killing trade frequency)
5. Looser entry thresholds to ensure >=80 trades train, >=12 trades test

Target: Sharpe > 0.612 (beat current best), trades >= 80 train, >= 12 test
Timeframe: 4h (20-50 trades/year target)
Size: 0.25-0.30 discrete levels
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_regime_1d1w_hma_atr_v1"
timeframe = "4h"
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
    """Connors RSI: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.zeros(n)
    streak = 0
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak = max(0, streak) + 1
        elif close[i] < close[i-1]:
            streak = min(0, streak) - 1
        else:
            streak = 0
        streak_rsi[i] = streak
    
    # Convert streak to RSI-like scale
    streak_rsi_scaled = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak_rsi[max(0, i-streak_period+1):i+1]
        if len(streak_window) >= streak_period:
            gain_streak = np.sum(np.where(streak_window > 0, streak_window, 0))
            loss_streak = np.sum(np.where(streak_window < 0, -streak_window, 0))
            if loss_streak > 1e-10:
                rs_streak = gain_streak / loss_streak
                streak_rsi_scaled[i] = 100.0 - (100.0 / (1.0 + rs_streak))
            else:
                streak_rsi_scaled[i] = 100.0
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1)
        percent_rank[i] = rank * 100.0
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi_scaled[i]):
            crsi[i] = (rsi_short[i] + streak_rsi_scaled[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures if market is chopping or trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        tr_sum = np.sum(tr[i-period+1:i+1])
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(highest_high - lowest_low) / np.log10(tr_sum)
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for ultra-macro filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=10)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # 4h HMA for trend confirmation
    hma_4h_fast = calculate_hma(close, period=16)
    hma_4h_slow = calculate_hma(close, period=48)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_fast[i]) or np.isnan(hma_4h_slow[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d + 1w HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i] and close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i] and close[i] < hma_1w_aligned[i]
        macro_neutral = not macro_bull and not macro_bear
        
        # === CHOPPINESS REGIME ===
        chop_range = chop[i] > 55.0  # Range/choppy market
        chop_trend = chop[i] < 45.0  # Trending market
        
        # === 4H TREND ===
        trend_bull = hma_4h_fast[i] > hma_4h_slow[i]
        trend_bear = hma_4h_fast[i] < hma_4h_slow[i]
        
        # === CONNORS RSI EXTREMES (mean reversion signals) ===
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        crsi_neutral = crsi[i] > 35.0 and crsi[i] < 65.0
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]
        breakout_short = close[i] < donchian_lower[i-1]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # LONG in RANGE regime: CRSI oversold + macro not bear
        if chop_range and crsi_oversold and not macro_bear:
            desired_signal = BASE_SIZE
        
        # LONG in TREND regime: breakout + macro bull + trend bull
        elif chop_trend and breakout_long and macro_bull and trend_bull:
            desired_signal = BASE_SIZE
        
        # LONG pullback in bull: CRSI neutral + macro bull + trend bull
        elif macro_bull and trend_bull and crsi_neutral and crsi[i] < 50.0:
            desired_signal = BASE_SIZE
        
        # SHORT in RANGE regime: CRSI overbought + macro not bull
        elif chop_range and crsi_overbought and not macro_bull:
            desired_signal = -BASE_SIZE
        
        # SHORT in TREND regime: breakout + macro bear + trend bear
        elif chop_trend and breakout_short and macro_bear and trend_bear:
            desired_signal = -BASE_SIZE
        
        # SHORT pullback in bear: CRSI neutral + macro bear + trend bear
        elif macro_bear and trend_bear and crsi_neutral and crsi[i] > 50.0:
            desired_signal = -BASE_SIZE
        
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
        if desired_signal > 0:
            final_signal = BASE_SIZE
        elif desired_signal < 0:
            final_signal = -BASE_SIZE
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