#!/usr/bin/env python3
"""
Experiment #1462: 12h Primary + 1d/1w HTF — Dual Regime with Relaxed Entries

Hypothesis: 12h timeframe with dual HTF filters will outperform 4h/1h in bear/range markets.
Based on research showing Choppiness Index regime detection works well (ETH Sharpe +0.923)
and Connors RSI mean reversion catches extremes in range markets.

Why 12h:
- 1h/4h strategies consistently failing in recent experiments (#1450-#1461 all negative/zero)
- 12h balances trade frequency (20-50/year) with signal quality
- Less noise than 4h, more trades than 1d

Design:
1. 1w HMA(21) = macro trend (bullish/bearish bias)
2. 1d HMA(21) = intermediate trend confirmation
3. Choppiness Index(14) = regime detection (>55 range, <45 trend)
4. Connors RSI = mean reversion signals in choppy regime
5. Donchian(20) = breakout signals in trending regime
6. ATR(14) 2.5x trailing stop = risk management
7. Position size 0.28 = conservative for 12h volatility

CRITICAL: Relaxed entry thresholds to ensure >=30 trades train, >=3 trades test
- CRSI: <25 oversold, >75 overbought (not <15/>85 which is too strict)
- CHOP: >55 range, <45 trend (wider bands for more regime switches)
- Donchian: close crosses previous bar high/low (simple breakout)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d1w_hma_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
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
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3"""
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak (consecutive up/down days)
    streak_rsi = np.full(n, np.nan)
    for i in range(streak_period, n):
        streak = 0
        if i > 0:
            if close[i] > close[i-1]:
                j = i
                while j > 0 and close[j] > close[j-1]:
                    streak += 1
                    j -= 1
            elif close[i] < close[i-1]:
                j = i
                while j > 0 and close[j] < close[j-1]:
                    streak -= 1
                    j -= 1
        # Convert streak to RSI-like value (0-100)
        if streak >= 0:
            streak_rsi[i] = min(100.0, streak * 50.0 / streak_period)
        else:
            streak_rsi[i] = max(0.0, 100.0 + streak * 50.0 / streak_period)
    
    # Percent Rank (100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        returns = np.diff(close[i-rank_period+1:i+1])
        if len(returns) > 0 and not np.any(np.isnan(returns)):
            current_return = returns[-1]
            count_below = np.sum(returns[:-1] < current_return)
            percent_rank[i] = 100.0 * count_below / (len(returns) - 1) if len(returns) > 1 else 50.0
    
    # Combine
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures market choppy vs trending"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        
        if highest_high > lowest_low:
            tr_sum = 0.0
            for j in range(i-period+1, i+1):
                tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            
            chop[i] = 100.0 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss sizing"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - breakout levels for entry trigger"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA for trend filters
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    donchian_20_upper, donchian_20_lower = calculate_donchian(high, low, period=20)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(donchian_20_upper[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1w HMA) - primary bias ===
        macro_bull = close[i] > hma_1w_aligned[i]
        macro_bear = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) - confirmation ===
        inter_bull = close[i] > hma_1d_aligned[i]
        inter_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        
        # === CONNORS RSI EXTREMES (mean reversion in chop) ===
        crsi_oversold = crsi[i] < 25.0  # Relaxed for more trades
        crsi_overbought = crsi[i] > 75.0  # Relaxed for more trades
        
        # === DONCHIAN BREAKOUT (trend following) ===
        breakout_long = close[i] > donchian_20_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_20_lower[i-1] if i > 0 else False
        
        # === DESIRED SIGNAL - DUAL REGIME LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # Path 1: Choppy regime + CRSI oversold + macro bull (mean reversion)
        if is_choppy and crsi_oversold and macro_bull:
            desired_signal = BASE_SIZE
        # Path 2: Trending regime + Donchian breakout + both HTF bull
        elif is_trending and breakout_long and macro_bull and inter_bull:
            desired_signal = BASE_SIZE
        # Path 3: Neutral regime + CRSI very oversold + macro bull
        elif crsi[i] < 20.0 and macro_bull:
            desired_signal = BASE_SIZE
        # Path 4: Simple long - macro bull + inter bull + CRSI < 35
        elif macro_bull and inter_bull and crsi[i] < 35.0:
            desired_signal = BASE_SIZE * 0.5
        
        # SHORT ENTRIES
        # Path 1: Choppy regime + CRSI overbought + macro bear (mean reversion)
        elif is_choppy and crsi_overbought and macro_bear:
            desired_signal = -BASE_SIZE
        # Path 2: Trending regime + Donchian breakout + both HTF bear
        elif is_trending and breakout_short and macro_bear and inter_bear:
            desired_signal = -BASE_SIZE
        # Path 3: Neutral regime + CRSI very overbought + macro bear
        elif crsi[i] > 80.0 and macro_bear:
            desired_signal = -BASE_SIZE
        # Path 4: Simple short - macro bear + inter bear + CRSI > 65
        elif macro_bear and inter_bear and crsi[i] > 65.0:
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
        if desired_signal >= BASE_SIZE * 0.4:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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
                # Flip position
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