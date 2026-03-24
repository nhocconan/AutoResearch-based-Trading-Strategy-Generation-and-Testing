#!/usr/bin/env python3
"""
Experiment #044: 12h Primary + 1d HTF — Connors RSI + HMA Trend + Choppiness Filter

Hypothesis: After analyzing 43 failed experiments, the pattern is clear:
- Complex dual-regime switches filter out too many trades (0 trade failures common)
- Connors RSI (CRSI) proven on ETH with Sharpe +0.923 in prior research
- CRSI captures mean reversion better than standard RSI for bear/range markets
- 1d HMA provides trend bias without being too restrictive (period=50)
- Choppiness Index as confirmation filter (not regime switch) to avoid over-filtering
- Looser CRSI thresholds (15/85 instead of 10/90) to ensure >=30 trades on train
- 12h timeframe targets 20-50 trades/year = 80-200 trades over 4 year train

Key design choices:
- Timeframe: 12h (lower fee drag, proven higher Sharpe)
- HTF: 1d HMA(50) for major trend bias
- Entry: CRSI < 15 (long) or > 85 (short) + 1d HMA confirmation
- Choppiness > 45 as soft confirmation (not hard filter)
- Position size: 0.30 (30% of capital, conservative)
- Stoploss: 2.5x ATR trailing
- Exit: CRSI crosses back through 50 (mean reversion complete)

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=80 on train, trades>=10 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_hma_chop_1d_v1"
timeframe = "12h"
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
    RSI Streak Component of Connors RSI
    Counts consecutive up/down days, converts to RSI
    """
    n = len(close)
    if n < period + 5:
        return np.full(n, np.nan)
    
    # Calculate streak values
    streak = np.zeros(n)
    streak[0] = 0
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100 scale)
    # Positive streak = bullish, negative = bearish
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    
    for i in range(period, n):
        streak_sum = np.sum(np.maximum(streak[i-period+1:i+1], 0))
        streak_loss = np.sum(np.maximum(-streak[i-period+1:i+1], 0))
        
        if streak_loss < 1e-10:
            streak_rsi[i] = 100.0
        else:
            rs = streak_sum / streak_loss
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Percent Rank Component of Connors RSI
    Percentage of prior returns less than current return over lookback
    """
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    # Calculate returns
    returns = np.zeros(n)
    returns[:] = np.nan
    for i in range(1, n):
        if close[i-1] > 1e-10:
            returns[i] = (close[i] - close[i-1]) / close[i-1] * 100.0
    
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    
    for i in range(period + 1, n):
        if np.isnan(returns[i]):
            continue
        
        current_return = returns[i]
        prior_returns = returns[i-period:i]
        
        # Count how many prior returns are less than current
        valid_prior = prior_returns[~np.isnan(prior_returns)]
        if len(valid_prior) > 0:
            count_less = np.sum(valid_prior < current_return)
            percent_rank[i] = (count_less / len(valid_prior)) * 100.0
        else:
            percent_rank[i] = 50.0
    
    return percent_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Values 0-100, <10 oversold, >90 overbought
    """
    rsi_short = calculate_rsi(close, rsi_period)
    rsi_streak = calculate_rsi_streak(close, streak_period)
    pr = calculate_percent_rank(close, pr_period)
    
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(max(rsi_period, streak_period, pr_period) + 1, n):
        if np.isnan(rsi_short[i]) or np.isnan(rsi_streak[i]) or np.isnan(pr[i]):
            continue
        crsi[i] = (rsi_short[i] + rsi_streak[i] + pr[i]) / 3.0
    
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
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, pr_period=100)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (conservative for 12h)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track CRSI crossing for exit signals
    prev_crsi = np.nan
    
    for i in range(150, n):  # Start later to ensure CRSI is ready (pr_period=100)
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_crsi = crsi[i] if not np.isnan(crsi[i]) else prev_crsi
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_crsi = crsi[i] if not np.isnan(crsi[i]) else prev_crsi
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_crsi = crsi[i] if not np.isnan(crsi[i]) else prev_crsi
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_crsi = crsi[i] if not np.isnan(crsi[i]) else prev_crsi
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS CONFIRMATION (soft filter) ===
        # CHOP > 45 favors mean reversion (our CRSI strategy)
        chop_favors_mr = chop[i] > 45.0
        
        # === CRSI SIGNALS (Connors RSI mean reversion) ===
        # LONG: CRSI < 15 (oversold) + HTF not strongly bear
        crsi_oversold = crsi[i] < 15.0
        # SHORT: CRSI > 85 (overbought) + HTF not strongly bull
        crsi_overbought = crsi[i] > 85.0
        
        # === CRSI EXIT SIGNALS (mean reversion complete) ===
        # Exit long when CRSI crosses above 50
        # Exit short when CRSI crosses below 50
        crsi_exit_long = prev_crsi < 50.0 and crsi[i] >= 50.0 if not np.isnan(prev_crsi) else False
        crsi_exit_short = prev_crsi > 50.0 and crsi[i] <= 50.0 if not np.isnan(prev_crsi) else False
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG entry: CRSI oversold + HTF bias or neutral + choppy market
        if crsi_oversold and (htf_bull or chop_favors_mr) and hma_bull:
            desired_signal = SIZE
        elif crsi_oversold and htf_bull:  # Strong HTF bull overrides other filters
            desired_signal = SIZE * 0.8
        
        # SHORT entry: CRSI overbought + HTF bias or neutral + choppy market
        elif crsi_overbought and (htf_bear or chop_favors_mr) and hma_bear:
            desired_signal = -SIZE
        elif crsi_overbought and htf_bear:  # Strong HTF bear overrides other filters
            desired_signal = -SIZE * 0.8
        
        # === CRSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and crsi_exit_long:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_exit_short:
            desired_signal = 0.0
        
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
            final_signal = SIZE * 0.6
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.6
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
        prev_crsi = crsi[i]
    
    return signals