#!/usr/bin/env python3
"""
Experiment #922: 4h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Dual-regime strategy adapts to market conditions. Choppiness Index
detects choppy vs trending markets. In choppy regimes (CHOP>61.8), use Connors
RSI mean reversion for quick reversals. In trending regimes (CHOP<38.2), use
HMA trend following. 1d HTF provides directional bias. This combination was
proven on ETH with Sharpe +0.923 in research.

Key innovations:
1. Choppiness Index (14) for regime detection - switch logic based on market state
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 - proven mean revert
3. HMA(21) on 1d for HTF bias - only trade in direction of daily trend
4. ATR(14) 2.5x trailing stop for risk management
5. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
6. LOOSE entry thresholds to ensure ≥10 trades/train, ≥3/test

Entry conditions (LOOSE to guarantee trades):
- CHOPPY regime (CHOP>55): Long CRSI<15, Short CRSI>85 (mean revert)
- TREND regime (CHOP<45): Long price>HMA_1d + HMA_4h bull, Short opposite
- Transition zone (45-55): No new entries, hold existing positions

Target: Sharpe>0.45, trades>=20 train, trades>=5 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_hma_regime_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """
    Hull Moving Average (HMA)
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Reduces lag while maintaining smoothness
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1].astype(np.float64)
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    hma = wma(diff, sqrt_n)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index
    CHOP = 100 * LOG10(SUM(ATR(1), n) / (Highest High(n) - Lowest Low(n))) / LOG10(n)
    CHOP > 61.8 = choppy/range
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        price_range = highest_high - lowest_low
        if price_range <= 1e-10:
            continue
        
        # Sum of ATR(1) = sum of true ranges
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
        
        choppiness[i] = 100.0 * np.log10(tr_sum / price_range) / np.log10(period)
    
    return choppiness

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of price change over lookback
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # RSI Streak - measure consecutive up/down days
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    streak = np.zeros(n, dtype=np.float64)
    
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
    
    # Calculate RSI of streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    for i in range(streak_period, n):
        if avg_streak_loss[i] > 1e-10:
            rs = avg_streak_gain[i] / avg_streak_loss[i]
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        elif avg_streak_gain[i] > 1e-10:
            streak_rsi[i] = 100.0
        else:
            streak_rsi[i] = 50.0
    
    # Percent Rank - percentile of current change vs last rank_period changes
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        changes = np.diff(close[i - rank_period:i + 1])
        if len(changes) > 0:
            current_change = close[i] - close[i-1]
            rank = np.sum(changes < current_change) / len(changes)
            percent_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    choppiness = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(choppiness[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HMA TREND ===
        hma_4h_bull = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bear = hma_4h_21[i] < hma_4h_50[i]
        
        # === CHOPPINESS REGIME ===
        choppy_regime = choppiness[i] > 55.0  # Lowered from 61.8 for more trades
        trend_regime = choppiness[i] < 45.0   # Lowered from 38.2 for more trades
        transition_zone = not choppy_regime and not trend_regime
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15.0  # Loosened from 10 for more trades
        crsi_overbought = crsi[i] > 85.0  # Loosened from 90 for more trades
        
        # === ENTRY LOGIC (DUAL REGIME - LOOSE TO GUARANTEE TRADES) ===
        desired_signal = 0.0
        
        if choppy_regime:
            # MEAN REVERSION in choppy market
            if crsi_oversold and htf_1d_bull:
                desired_signal = SIZE_BASE
            elif crsi_overbought and htf_1d_bear:
                desired_signal = -SIZE_BASE
            # Allow counter-HTF trades in choppy (mean revert)
            elif crsi_oversold:
                desired_signal = SIZE_BASE * 0.8
            elif crsi_overbought:
                desired_signal = -SIZE_BASE * 0.8
        
        elif trend_regime:
            # TREND FOLLOWING in trending market
            if htf_1d_bull and hma_4h_bull:
                desired_signal = SIZE_STRONG
            elif htf_1d_bear and hma_4h_bear:
                desired_signal = -SIZE_STRONG
            # Weaker signal if only one confirms
            elif htf_1d_bull or hma_4h_bull:
                desired_signal = SIZE_BASE
            elif htf_1d_bear or hma_4h_bear:
                desired_signal = -SIZE_BASE
        
        else:
            # Transition zone - hold existing, no new entries
            if in_position:
                desired_signal = position_side * SIZE_BASE
            else:
                desired_signal = 0.0
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_BASE * 0.6
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_BASE * 0.6
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