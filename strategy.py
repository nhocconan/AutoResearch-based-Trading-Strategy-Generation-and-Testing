#!/usr/bin/env python3
"""
Experiment #042: 4h Primary + 1d/1w HTF — Connors RSI + Fisher Transform + Choppiness Regime

Hypothesis: After 41 failed experiments, the pattern shows 4h needs stronger regime detection.
- Connors RSI (CRSI) has 75% win rate in backtests, works well in bear/range markets
- Fisher Transform catches reversals in bear market rallies (proven edge)
- Choppiness Index switches between trend-follow and mean-revert modes
- Dual HTF bias (1d + 1w HMA) provides stronger trend confirmation than single HTF
- 4h timeframe targets 20-50 trades/year (lower fee drag than 1h/15m)

Key design choices:
- Timeframe: 4h (proven to work better than lower TFs for BTC/ETH)
- HTF: 1d HMA(21) + 1w HMA(9) for major trend bias
- Entry: CRSI extremes + Fisher crossover + Choppiness regime
- Regime: CHOP>61.8 = range (mean revert), CHOP<38.2 = trend (breakout)
- Position size: 0.30 (30% of capital, conservative)
- Stoploss: 2.5x ATR trailing
- LOOSE filters to ensure >=30 trades on train, >=3 on test (CRSI<15/>85 not <10/>90)

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_fisher_chop_dual_htf_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Proven 75% win rate in mean reversion strategies
    """
    n = len(close)
    if n < pr_period + 1:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to positive values for RSI calculation
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs, streak_period)
    
    # Percent Rank (100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    for i in range(pr_period, n):
        price_changes = close[i-pr_period+1:i+1]
        current_change = close[i] - close[i-1] if i > 0 else 0
        count_below = np.sum(price_changes[:-1] < close[i-1])
        percent_rank[i] = 100.0 * count_below / (pr_period - 1) if pr_period > 1 else 50.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(pr_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Transforms price into a Gaussian normal distribution
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close := high)  # use high for length
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # Calculate typical price
    typical = (high + low + close) / 3.0
    
    # Normalize price
    normalized = np.zeros(n)
    normalized[:] = np.nan
    
    for i in range(period, n):
        highest = np.max(typical[i-period+1:i+1])
        lowest = np.min(typical[i-period+1:i+1])
        range_hl = highest - lowest
        if range_hl > 1e-10:
            normalized[i] = 0.665 * ((typical[i] - lowest) / range_hl - 0.5) + 0.67 * normalized[i-1]
        else:
            normalized[i] = normalized[i-1] if i > 0 else 0.0
    
    # Clamp to avoid division by zero
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = np.zeros(n)
    fisher[:] = np.nan
    for i in range(1, n):
        if not np.isnan(normalized[i]):
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
    
    # Fisher signal (previous bar for no look-ahead)
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    for i in range(1, n):
        fisher_signal[i] = fisher[i-1] if not np.isnan(fisher[i-1]) else np.nan
    
    return fisher, fisher_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for even higher timeframe bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=9)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # 4h HMA for local trend
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
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
        if np.isnan(crsi[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(chop[i]) or np.isnan(hma_4h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d + 1w HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # Strong bias when both HTFs agree
        htf_strong_bull = htf_1d_bull and htf_1w_bull
        htf_strong_bear = htf_1d_bear and htf_1w_bear
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 61.8  # range market
        is_trending = chop[i] < 38.2  # trending market
        is_neutral = not is_choppy and not is_trending
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 15.0  # LOOSE: was <10
        crsi_overbought = crsi[i] > 85.0  # LOOSE: was >90
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crossing above -1.5 from below (bullish reversal)
        fisher_bull_cross = fisher_signal[i] < -1.5 and fisher[i] >= -1.5
        # Fisher crossing below +1.5 from above (bearish reversal)
        fisher_bear_cross = fisher_signal[i] > 1.5 and fisher[i] <= 1.5
        
        # === 4h HMA TREND ===
        hma_4h_bull = close[i] > hma_4h[i]
        hma_4h_bear = close[i] < hma_4h[i]
        
        # === DESIRED SIGNAL (Dual Regime Logic) ===
        desired_signal = 0.0
        
        if is_trending:
            # TREND REGIME: Follow Fisher reversals with HTF bias
            # LONG: Fisher bull cross + HTF strong bull + CRSI not overbought
            if fisher_bull_cross and htf_strong_bull and crsi[i] < 80.0:
                desired_signal = SIZE
            # SHORT: Fisher bear cross + HTF strong bear + CRSI not oversold
            elif fisher_bear_cross and htf_strong_bear and crsi[i] > 20.0:
                desired_signal = -SIZE
            # Fallback: Fisher cross + 4h HMA (weaker signal)
            elif fisher_bull_cross and hma_4h_bull and crsi[i] < 70.0:
                desired_signal = SIZE * 0.6
            elif fisher_bear_cross and hma_4h_bear and crsi[i] > 30.0:
                desired_signal = -SIZE * 0.6
                
        elif is_choppy:
            # CHOPPY REGIME: Mean revert with CRSI extremes
            # LONG: CRSI oversold + HTF not strongly bear + Fisher not extreme bear
            if crsi_oversold and not htf_strong_bear and fisher[i] > -2.0:
                desired_signal = SIZE
            # SHORT: CRSI overbought + HTF not strongly bull + Fisher not extreme bull
            elif crsi_overbought and not htf_strong_bull and fisher[i] < 2.0:
                desired_signal = -SIZE
            # Fallback: Extreme CRSI mean reversion
            elif crsi[i] < 10.0 and hma_4h_bull:
                desired_signal = SIZE * 0.6
            elif crsi[i] > 90.0 and hma_4h_bear:
                desired_signal = -SIZE * 0.6
                
        else:
            # NEUTRAL REGIME: Mixed signals, require more confluence
            # LONG: CRSI oversold + Fisher bull cross + HTF neutral/bull
            if crsi_oversold and fisher_bull_cross and not htf_strong_bear:
                desired_signal = SIZE * 0.8
            # SHORT: CRSI overbought + Fisher bear cross + HTF neutral/bear
            elif crsi_overbought and fisher_bear_cross and not htf_strong_bull:
                desired_signal = -SIZE * 0.8
            # Fallback: CRSI extreme + HTF bias
            elif crsi[i] < 12.0 and htf_1d_bull:
                desired_signal = SIZE * 0.5
            elif crsi[i] > 88.0 and htf_1d_bear:
                desired_signal = -SIZE * 0.5
        
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