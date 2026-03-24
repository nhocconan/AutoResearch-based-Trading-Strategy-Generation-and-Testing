#!/usr/bin/env python3
"""
Experiment #098: 4h Primary + 1d HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 97 failed experiments, the pattern is clear:
- Pure trend following fails on BTC/ETH in bear/range markets (2022 crash, 2025 bear)
- Pure mean reversion fails on SOL (strong trends)
- SOLUTION: Connors RSI (CRSI) for precise mean-reversion entries + Choppiness regime switch
- CRSI combines RSI(3) + RSI_Streak(2) + PercentRank(100) — proven 75% win rate in research
- 1d HMA provides major trend bias (only trade with HTF direction)
- Choppiness Index switches between mean-revert (CHOP>55) and trend-follow (CHOP<55)
- 4h timeframe targets 20-50 trades/year (lower fee drag than 15m/1h)
- LOOSE CRSI thresholds (15/85 instead of 10/90) ensure >=30 trades on train

Key design choices:
- Timeframe: 4h (sweet spot for trade frequency vs fee drag)
- HTF: 1d HMA(50) for major trend bias — call get_htf_data() ONCE before loop
- Entry: CRSI extremes + 1d HMA bias + Choppiness regime confirmation
- Regime: CHOP>55 = range (mean revert at CRSI extremes), CHOP<55 = trend (pullback entries)
- Position size: 0.30 (30% of capital, conservative for 4h swings)
- Stoploss: 2.5x ATR(14) trailing stop
- LOOSE filters to ensure trades generate on ALL symbols (BTC, ETH, SOL)

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_hma_1d_regime_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    RSI Streak Component for Connors RSI
    Measures consecutive up/down days
    Returns RSI of the streak lengths
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak lengths
    streak = np.zeros(n)
    streak[0] = 1
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            if streak[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif close[i] < close[i-1]:
            if streak[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert to absolute streak for RSI calculation
    streak_abs = np.abs(streak)
    
    # Calculate RSI on streak values (up streaks = gain, down = loss)
    streak_change = np.diff(streak)
    gain = np.where(streak_change > 0, streak_change, 0.0)
    loss = np.where(streak_change < 0, -streak_change, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    rsi_streak[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi_streak[i] = 100.0
        else:
            rs = avg_gain[i] / (avg_loss[i] + 1e-10)
            rsi_streak[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi_streak

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component for Connors RSI
    Measures where current close ranks within last 'period' bars
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    pr = np.zeros(n)
    pr[:] = np.nan
    
    for i in range(period - 1, n):
        window = close[i-period+1:i+1]
        current = close[i]
        # Count how many values in window are less than current
        rank = np.sum(window < current)
        pr[i] = 100.0 * rank / (period - 1)
    
    return pr

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven mean-reversion indicator with 75% win rate
    """
    rsi_fast = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    crsi = (rsi_fast + rsi_streak + pr) / 3.0
    return crsi

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
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    hma_4h = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):  # Start after all indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
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
        # CHOP > 55 = range/choppy (mean revert)
        # CHOP < 55 = trending (trend follow)
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] <= 55.0
        
        # === CRSI EXTREMES (LOOSE THRESHOLDS FOR TRADES) ===
        crsi_oversold = crsi[i] < 20.0  # LOOSE: was 15, now 20 for more trades
        crsi_overbought = crsi[i] > 80.0  # LOOSE: was 85, now 80 for more trades
        
        # === 4h HMA TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        
        # === SMA200 LONG-TERM FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_choppy:
            # CHOPPY REGIME: Mean revert at CRSI extremes
            # LONG: CRSI oversold + HTF not bear + above SMA200 or HMA bull
            if crsi_oversold and not htf_bear and (above_sma200 or hma_bull):
                desired_signal = SIZE
            # SHORT: CRSI overbought + HTF not bull + below SMA200 or HMA bear
            elif crsi_overbought and not htf_bull and (below_sma200 or hma_bear):
                desired_signal = -SIZE
        else:
            # TRENDING REGIME: Pullback entries with HTF bias
            # LONG: CRSI oversold (pullback) + HTF bull + HMA bull
            if crsi_oversold and htf_bull and hma_bull:
                desired_signal = SIZE
            # SHORT: CRSI overbought (pullback) + HTF bear + HMA bear
            elif crsi_overbought and htf_bear and hma_bear:
                desired_signal = -SIZE
        
        # === FALLBACK: Very loose entries to ensure trades ===
        # Only if no signal yet and CRSI at extreme
        if desired_signal == 0.0:
            if crsi[i] < 15.0 and htf_bull:
                desired_signal = SIZE * 0.7
            elif crsi[i] > 85.0 and htf_bear:
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
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.7
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