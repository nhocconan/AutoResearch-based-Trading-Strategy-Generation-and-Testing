#!/usr/bin/env python3
"""
Experiment #1356: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime

Hypothesis: #1352 (12h Donchian+HMA+RSI) achieved Sharpe=0.571, but used standard RSI
which is slow. Connors RSI (CRSI) has 75% win rate for mean-reversion entries.
Combined with Choppiness Index regime filter, this should:
1. Enter on extreme CRSI (<10 long, >90 short) for better timing
2. Use CHOP to avoid trend signals in choppy markets (CHOP>61.8)
3. Use 1d KAMA for adaptive macro trend (better than HMA in ranges)
4. Maintain 12h KAMA for primary trend with faster response
5. Target 20-50 trades/year with 0.28 position size

Key improvements over #1352:
- CRSI instead of RSI (faster mean-reversion signals)
- Choppiness filter (avoid false breakouts in range)
- KAMA instead of HMA (adapts to volatility)
- Dual entry: trend breakout OR mean-reversion pullback

Target: Sharpe > 0.618, trades >= 30 train, >= 5 test, DD < -40%
Timeframe: 12h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_kama_1d_atr_regime_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency"""
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio
    er = np.full(n, np.nan)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate smoothing constant
    sc = np.full(n, np.nan)
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Initialize KAMA
    kama[period] = close[period]
    
    # Calculate KAMA
    for i in range(period + 1, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i - 1]):
            kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI - (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_rsi3 = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    loss_rsi3 = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi3 = np.full(n, np.nan)
    mask = loss_rsi3 > 1e-10
    rsi3[mask] = 100.0 - (100.0 / (1.0 + gain_rsi3[mask] / loss_rsi3[mask]))
    rsi3[loss_rsi3 <= 1e-10] = 100.0
    rsi3[:rsi_period] = np.nan
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak = 0
        if close[i] >= close[i - 1]:
            for j in range(i, max(0, i - 20), -1):
                if j == 0 or close[j] < close[j - 1]:
                    break
                streak += 1
        else:
            for j in range(i, max(0, i - 20), -1):
                if j == 0 or close[j] >= close[j - 1]:
                    break
                streak -= 1
        
        # Convert streak to RSI-like value
        abs_streak = abs(streak)
        if abs_streak == 0:
            streak_rsi[i] = 50.0
        else:
            streak_delta = np.zeros(abs_streak + 1)
            if streak > 0:
                streak_delta[1:] = 1.0
            else:
                streak_delta[1:] = -1.0
            
            streak_gain = np.where(streak_delta > 0, streak_delta, 0.0)
            streak_loss = np.where(streak_delta < 0, -streak_delta, 0.0)
            
            avg_gain = np.mean(streak_gain) if len(streak_gain) > 0 else 0.0
            avg_loss = np.mean(streak_loss) if len(streak_loss) > 0 else 0.0
            
            if avg_loss > 1e-10:
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
            else:
                streak_rsi[i] = 100.0
    
    streak_rsi[:streak_period] = np.nan
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        if not np.any(np.isnan(window)):
            count_below = np.sum(window[:-1] < window[-1])
            percent_rank[i] = (count_below / (rank_period - 1)) * 100.0
    
    # Combine into CRSI
    for i in range(rank_period, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppiness vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest = np.nanmax(high[i - period + 1:i + 1])
        lowest = np.nanmin(low[i - period + 1:i + 1])
        
        if highest > lowest:
            atr_sum = 0.0
            for j in range(i - period + 1, i + 1):
                if j > 0:
                    tr = max(high[j] - low[j], abs(high[j] - close[j - 1]), abs(low[j] - close[j - 1]))
                    atr_sum += tr
            
            if atr_sum > 1e-10:
                chop[i] = 100.0 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i - period + 1:i + 1])
        lower[i] = np.nanmin(low[i - period + 1:i + 1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF KAMA for macro trend
    kama_1d_raw = calculate_kama(df_1d['close'].values, period=10)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate primary (12h) indicators
    kama_12h = calculate_kama(close, period=10)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    atr = calculate_atr(high, low, close, period=14)
    
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
        if np.isnan(kama_12h[i]) or np.isnan(kama_1d_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (1d KAMA) ===
        macro_bull = close[i] > kama_1d_aligned[i]
        macro_bear = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (12h KAMA) ===
        trend_bull = close[i] > kama_12h[i]
        trend_bear = close[i] < kama_12h[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8  # Range market
        is_trending = chop[i] < 38.2  # Trending market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Strong mean-reversion long signal
        crsi_overbought = crsi[i] > 85.0  # Strong mean-reversion short signal
        crsi_neutral_bull = crsi[i] > 45.0 and crsi[i] < 60.0  # Pullback entry long
        crsi_neutral_bear = crsi[i] < 55.0 and crsi[i] > 40.0  # Pullback entry short
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i - 1]
        breakout_short = close[i] < donchian_lower[i - 1]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY - Multiple paths
        # Path 1: Trending market + Donchian breakout + trend confirmation
        if is_trending and breakout_long and trend_bull and macro_bull:
            desired_signal = BASE_SIZE
        # Path 2: Choppy market + CRSI oversold (mean reversion)
        elif is_choppy and crsi_oversold and close[i] > kama_12h[i]:
            desired_signal = BASE_SIZE * 0.7
        # Path 3: CRSI pullback in uptrend (buy the dip)
        elif crsi_neutral_bull and trend_bull and macro_bull:
            desired_signal = BASE_SIZE * 0.5
        # Path 4: Simple trend follow (both KAMAs bullish)
        elif trend_bull and macro_bull and not is_choppy:
            desired_signal = BASE_SIZE * 0.4
        
        # SHORT ENTRY - Multiple paths
        # Path 1: Trending market + Donchian breakdown + trend confirmation
        elif is_trending and breakout_short and trend_bear and macro_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Choppy market + CRSI overbought (mean reversion)
        elif is_choppy and crsi_overbought and close[i] < kama_12h[i]:
            desired_signal = -BASE_SIZE * 0.7
        # Path 3: CRSI pullback in downtrend (sell the rip)
        elif crsi_neutral_bear and trend_bear and macro_bear:
            desired_signal = -BASE_SIZE * 0.5
        # Path 4: Simple trend follow (both KAMAs bearish)
        elif trend_bear and macro_bear and not is_choppy:
            desired_signal = -BASE_SIZE * 0.4
        
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