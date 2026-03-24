#!/usr/bin/env python3
"""
Experiment #151: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + Donchian Breakout

Hypothesis: After 150 experiments, the clearest pattern is:
- Pure trend following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
- Pure mean reversion fails on SOL (strong trends)
- BEST SOLUTION: Dual-regime with Choppiness Index to switch between trend/mean-revert
- Connors RSI (CRSI) has 75% win rate for mean reversion entries
- Donchian(20) breakouts work for trend following on SOL
- 1d HMA provides major trend bias without being too restrictive
- 4h timeframe targets 20-50 trades/year (optimal fee/trade balance)

Key design choices:
- Timeframe: 4h (proven best balance of signal quality vs trade frequency)
- HTF: 1d HMA(50) for major trend bias
- Regime: CHOP>55 = range (Connors RSI mean revert), CHOP<55 = trend (Donchian breakout)
- Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
- Entry: CRSI<10 long / CRSI>90 short in choppy regime
- Entry: Donchian breakout + HTF bias in trending regime
- Position size: 0.28 (28% of capital, conservative)
- Stoploss: 2.5x ATR trailing

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_donchian_hma_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures market choppiness vs trending
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        sum_tr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10 and sum_tr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_tr / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_crsi(close):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - streak of consecutive up/down days
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    streak = 0
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak = max(0, streak) + 1
        elif close[i] < close[i-1]:
            streak = min(0, streak) - 1
        else:
            streak = 0
        
        # Convert streak to RSI-like value (0-100)
        if streak > 0:
            streak_rsi[i] = 50.0 + min(streak * 10, 50)
        elif streak < 0:
            streak_rsi[i] = 50.0 - min(abs(streak) * 10, 50)
        else:
            streak_rsi[i] = 50.0
    
    # Apply RSI(2) to streak values
    streak_rsi2 = calculate_rsi(streak_rsi, period=2)
    
    # Percent Rank (100) - percentage of prior 100 days with lower returns
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(100, n):
        returns = np.diff(close[i-100:i+1])
        if len(returns) > 0:
            current_return = returns[-1]
            count_lower = np.sum(returns[:-1] < current_return)
            percent_rank[i] = 100.0 * count_lower / (len(returns) - 1)
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(100, n):
        if not np.isnan(rsi3[i]) and not np.isnan(streak_rsi2[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi3[i] + streak_rsi2[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    crsi = calculate_crsi(close)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 4h)
    
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
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range/choppy (mean revert with CRSI)
        # CHOP < 55 = trending (breakout follow with Donchian)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === CONNORS RSI SIGNALS (Mean Reversion in Choppy Regime) ===
        crsi_oversold = crsi[i] < 15.0  # Very oversold
        crsi_overbought = crsi[i] > 85.0  # Very overbought
        
        # === DONCHIAN BREAKOUT SIGNALS (Trend Following in Trending Regime) ===
        donchian_breakout_bull = close[i] > donchian_upper[i-1]
        donchian_breakout_bear = close[i] < donchian_lower[i-1]
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === RSI FILTER (Loose to ensure trades) ===
        rsi_ok_long = rsi[i] > 25.0
        rsi_ok_short = rsi[i] < 75.0
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_choppy:
            # CHOPPY REGIME: Connors RSI Mean Reversion
            # LONG: CRSI oversold + HTF not strongly bear + RSI ok
            if crsi_oversold and not htf_bear and rsi_ok_long:
                desired_signal = SIZE
            # SHORT: CRSI overbought + HTF not strongly bull + RSI ok
            elif crsi_overbought and not htf_bull and rsi_ok_short:
                desired_signal = -SIZE
            # Fallback: Extreme CRSI with HMA confirmation
            elif crsi[i] < 10.0 and hma_bull:
                desired_signal = SIZE * 0.7
            elif crsi[i] > 90.0 and hma_bear:
                desired_signal = -SIZE * 0.7
        else:
            # TRENDING REGIME: Donchian Breakout with HTF Bias
            # LONG: breakout + HTF bull + HMA bull
            if donchian_breakout_bull and htf_bull and hma_bull:
                desired_signal = SIZE
            # SHORT: breakout + HTF bear + HMA bear
            elif donchian_breakout_bear and htf_bear and hma_bear:
                desired_signal = -SIZE
            # Fallback: Breakout with HMA only (stronger signal)
            elif donchian_breakout_bull and hma_bull and rsi[i] > 35.0:
                desired_signal = SIZE * 0.7
            elif donchian_breakout_bear and hma_bear and rsi[i] < 65.0:
                desired_signal = -SIZE * 0.7
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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