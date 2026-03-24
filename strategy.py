#!/usr/bin/env python3
"""
Experiment #926: 1d Primary + 1w HTF — Dual Regime (Chopiness + Connors RSI + HMA)

Hypothesis: Daily timeframe with weekly HTF bias captures major moves while avoiding
whipsaw. Choppiness Index detects regime (chop vs trend), then applies appropriate
strategy: Connors RSI for mean reversion in chop, HMA trend following in trends.
Weekly HMA(21) provides macro bias filter. This dual-regime approach should work
across BTC/ETH/SOL in both bull and bear markets.

Key innovations:
1. Choppiness Index(14) regime detection: >50 = chop (mean revert), <50 = trend
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 for precise MR entries
3. 1w HMA(21) for HTF macro bias (price above = long bias, below = short bias)
4. 1d HMA(16/48) for trend confirmation
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30 to minimize fee churn
7. LOOSE entry thresholds to guarantee ≥10 trades/train, ≥3/test

Entry conditions:
- CHOPPY regime (CHOP>50): Long when CRSI<15 + price>1w_HMA, Short when CRSI>85 + price<1w_HMA
- TREND regime (CHOP<50): Long when 1d_HMA16>48 + price>1w_HMA, Short when 1d_HMA16<48 + price<1w_HMA
- RSI filter only avoids extremes (not strict)

Target: Sharpe>0.45, trades>=20 train, trades>=5 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_chop_crsi_hma_1w_v1"
timeframe = "1d"
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
        result = np.full(len(series), np.nan)
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures if market is choppy (high CHOP) or trending (low CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/ranging, CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    chop = np.full(n, np.nan, dtype=np.float64)
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        range_val = highest_high - lowest_low
        
        if range_val > 1e-10 and sum_atr > 1e-10:
            chop[i] = 100.0 * np.log10(sum_atr / range_val) / np.log10(period)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Short-term momentum
    RSI(Streak): Consecutive up/down days momentum
    PercentRank: Where current price ranks vs last 100 days
    
    CRSI < 10 = oversold (long), CRSI > 90 = overbought (short)
    """
    n = len(close)
    if n < rank_period:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak (convert to absolute for RSI calc)
    streak_abs = np.abs(streak)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    if n >= streak_period + 1:
        delta = np.diff(streak_abs, prepend=streak_abs[0])
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        
        avg_gain = pd.Series(gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
        
        for i in range(streak_period, n):
            if avg_loss[i] > 1e-10:
                rs = avg_gain[i] / avg_loss[i]
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            else:
                streak_rsi[i] = 100.0
    
    # PercentRank
    percent_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        count_below = np.sum(window[:-1] < current)
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d_16 = calculate_hma(close, period=16)
    hma_1d_48 = calculate_hma(close, period=48)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    
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
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_16[i]) or np.isnan(hma_1d_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 50.0  # >50 = choppy/ranging, <50 = trending
        is_trending = chop_14[i] <= 50.0
        
        # === 1d HMA TREND ===
        hma_1d_bull = hma_1d_16[i] > hma_1d_48[i]
        hma_1d_bear = hma_1d_16[i] < hma_1d_48[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20.0  # Loose threshold for more trades
        crsi_overbought = crsi[i] > 80.0  # Loose threshold for more trades
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        desired_signal = 0.0
        
        if is_choppy:
            # MEAN REVERSION regime - use Connors RSI extremes
            if htf_1w_bull and crsi_oversold:
                desired_signal = SIZE_BASE  # Long on oversold in bull HTF
            elif htf_1w_bear and crsi_overbought:
                desired_signal = -SIZE_BASE  # Short on overbought in bear HTF
        else:
            # TREND regime - use HMA crossover/alignment
            if htf_1w_bull and hma_1d_bull:
                desired_signal = SIZE_BASE  # Long in bull HTF + bull 1d
            elif htf_1w_bear and hma_1d_bear:
                desired_signal = -SIZE_BASE  # Short in bear HTF + bear 1d
        
        # === HMA CROSSOVER ENTRIES (additional trigger for more trades) ===
        hma_crossover_long = False
        hma_crossover_short = False
        if i > 0 and not np.isnan(hma_1d_16[i-1]) and not np.isnan(hma_1d_48[i-1]):
            hma_crossover_long = (hma_1d_16[i-1] <= hma_1d_48[i-1]) and (hma_1d_16[i] > hma_1d_48[i])
            hma_crossover_short = (hma_1d_16[i-1] >= hma_1d_48[i-1]) and (hma_1d_16[i] < hma_1d_48[i])
        
        # Add crossover entries for more trade frequency
        if hma_crossover_long and htf_1w_bull:
            desired_signal = max(desired_signal, SIZE_STRONG)
        elif hma_crossover_short and htf_1w_bear:
            desired_signal = min(desired_signal, -SIZE_STRONG)
        
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