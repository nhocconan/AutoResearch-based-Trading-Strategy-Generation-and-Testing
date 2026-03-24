#!/usr/bin/env python3
"""
Experiment #988: 4h Primary + 12h/1d HTF — Dual Regime (CHOP + CRSI/KAMA)

Hypothesis: 4h timeframe with Choppiness Index regime detection + dual entry logic
will capture both trend and mean-reversion opportunities while HTF bias filters
counter-trend trades. This targets the proven ETH pattern (CHOP+CRSI Sharpe +0.923).

Key innovations:
1. 12h HMA(21) for intermediate trend bias (aligns with multi-day swings)
2. 1d momentum (close > SMA50) for long-term direction
3. CHOP(14) regime: <38.2 = trend (use KAMA crossover), >61.8 = range (use CRSI)
4. Connors RSI for mean-reversion entries in choppy markets
5. KAMA(10/30) for adaptive trend entries (responds to volatility changes)
6. ATR(14) 2.5x trailing stop for risk management

Why this should work on 4h:
- 4h captures 2-5 day swings, ideal for crypto volatility
- CHOP filter avoids whipsaws in 2022 bottom and 2025 range
- CRSI has 75% win rate for oversold/overbought reversals
- HTF bias (12h/1d) prevents fighting the macro trend
- KAMA adapts to volatility, better than fixed EMA in crypto

Entry conditions (BALANCED for trade frequency):
- LONG = 12h bull + 1d bull + (CHOP<38 + KAMA cross OR CHOP>61 + CRSI<18)
- SHORT = 12h bear + 1d bear + (CHOP<38 + KAMA cross OR CHOP>61 + CRSI>82)
- Relaxed thresholds to ensure 30+ trades/train, 3+/test

Target: Sharpe>0.45, trades>=30 train, trades>=3 test, DD>-40%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_crsi_kama_regime_12h1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, fast_period=2, slow_period=30, er_period=10):
    """Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < er_period + 1:
        return np.full(n, np.nan)
    
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    kama = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(er_period, n):
        net_change = abs(close[i] - close[i - er_period])
        total_noise = 0.0
        for j in range(i - er_period + 1, i + 1):
            total_noise += abs(close[j] - close[j - 1])
        
        er = net_change / total_noise if total_noise > 1e-10 else 0.0
        sc = er * (fast_sc - slow_sc) + slow_sc
        
        if i == er_period:
            kama[i] = close[i]
        else:
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_hma(close, period):
    """Hull Moving Average"""
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
    
    return wma(diff, sqrt_n)

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
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    Measures whether market is trending or choppy/ranging
    CHOP > 61.8 = range/choppy
    CHOP < 38.2 = trending
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr_sum += max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
        
        if highest_high > lowest_low and tr_sum > 1e-10:
            chop[i] = 100.0 * np.log10((highest_high - lowest_low) / tr_sum) / np.log10(period)
    
    return chop

def calculate_crsi(close):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of daily returns over 100 days
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2)
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(2, n):
        abs_streak = abs(streak[i])
        if abs_streak >= 2:
            streak_rsi[i] = 100.0 if streak[i] > 0 else 0.0
        elif abs_streak == 1:
            streak_rsi[i] = 75.0 if streak[i] > 0 else 25.0
        else:
            streak_rsi[i] = 50.0
    
    # PercentRank(100) - percentile of daily return in last 100 days
    returns = np.diff(close, prepend=close[0]) / (np.abs(close) + 1e-10)
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        window = returns[i-99:i+1]
        pct_rank[i] = 100.0 * np.sum(returns[i] >= window) / len(window)
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Daily SMA50 for long-term trend
    sma_1d_raw = calculate_sma(df_1d['close'].values, period=50)
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_raw)
    
    # Calculate 4h indicators
    kama_4h_10 = calculate_kama(close, fast_period=2, slow_period=10, er_period=10)
    kama_4h_30 = calculate_kama(close, fast_period=2, slow_period=30, er_period=10)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close)
    
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
        
        if np.isnan(kama_4h_10[i]) or np.isnan(kama_4h_30[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h_aligned[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA + 1d SMA50) ===
        htf_12h_bull = close[i] > hma_12h_aligned[i]
        htf_12h_bear = close[i] < hma_12h_aligned[i]
        
        htf_1d_bull = close[i] > sma_1d_aligned[i]
        htf_1d_bear = close[i] < sma_1d_aligned[i]
        
        # === REGIME DETECTION (CHOP) ===
        is_trending = chop_14[i] < 38.2
        is_ranging = chop_14[i] > 61.8
        is_neutral = not is_trending and not is_ranging
        
        # === 4h KAMA CROSSOVER ===
        kama_crossover_long = False
        kama_crossover_short = False
        if i > 0 and not np.isnan(kama_4h_10[i-1]) and not np.isnan(kama_4h_30[i-1]):
            kama_crossover_long = (kama_4h_10[i-1] <= kama_4h_30[i-1]) and (kama_4h_10[i] > kama_4h_30[i])
            kama_crossover_short = (kama_4h_10[i-1] >= kama_4h_30[i-1]) and (kama_4h_10[i] < kama_4h_30[i])
        
        # === KAMA TREND CONTINUATION ===
        kama_bull_trend = kama_4h_10[i] > kama_4h_30[i]
        kama_bear_trend = kama_4h_10[i] < kama_4h_30[i]
        
        # === CRSI EXTREMES (RELAXED FOR MORE TRADES) ===
        crsi_oversold = crsi[i] < 18  # Relaxed from 10
        crsi_overbought = crsi[i] > 82  # Relaxed from 90
        crsi_moderate_oversold = crsi[i] < 35
        crsi_moderate_overbought = crsi[i] > 65
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # LONG entries
        if htf_12h_bull and htf_1d_bull:
            if is_trending and kama_crossover_long:
                # Trend regime: KAMA crossover entry (strong signal)
                desired_signal = SIZE_STRONG
            elif is_ranging and crsi_oversold:
                # Range regime: CRSI mean reversion entry
                desired_signal = SIZE_BASE
            elif is_neutral and kama_bull_trend and crsi_moderate_oversold:
                # Neutral regime: trend continuation with pullback
                desired_signal = SIZE_BASE
            elif kama_bull_trend and crsi[i] < 45:
                # Additional long condition for trade frequency
                desired_signal = SIZE_BASE
        
        # SHORT entries
        elif htf_12h_bear and htf_1d_bear:
            if is_trending and kama_crossover_short:
                # Trend regime: KAMA crossover entry (strong signal)
                desired_signal = -SIZE_STRONG
            elif is_ranging and crsi_overbought:
                # Range regime: CRSI mean reversion entry
                desired_signal = -SIZE_BASE
            elif is_neutral and kama_bear_trend and crsi_moderate_overbought:
                # Neutral regime: trend continuation with pullback
                desired_signal = -SIZE_BASE
            elif kama_bear_trend and crsi[i] > 55:
                # Additional short condition for trade frequency
                desired_signal = -SIZE_BASE
        
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