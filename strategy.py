#!/usr/bin/env python3
"""
Experiment #734: 1d Primary + 1w HTF — Connors RSI + Choppiness Regime Switch

Hypothesis: Daily timeframe with weekly trend bias provides optimal signal quality
for crypto perpetuals. Connors RSI (CRSI) has proven 75% win rate in bear markets,
and Choppiness Index effectively switches between mean-reversion (chop) and
trend-follow (trending) regimes. This combination should work through 2022 crash
and 2025 bear market.

Key innovations:
1. 1w HMA(21) for primary trend bias — smooth, lag-reduced
2. Connors RSI(3,2,100) for entries — proven mean-reversion edge
3. Choppiness Index(14) regime filter — >61.8=range, <38.2=trend
4. Dual-mode logic: mean-revert in chop, trend-follow otherwise
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.25, ±0.30

Entry conditions (LOOSE to ensure ≥20 trades/train, ≥3/test):
- LONG in CHOP: CRSI<15 + price>SMA200
- LONG in TREND: CRSI<30 + 1w HMA bull + price>1d HMA
- SHORT in CHOP: CRSI>85 + price<SMA200
- SHORT in TREND: CRSI>70 + 1w HMA bear + price<1d HMA

Target: Sharpe>0.40, trades>=20 train, trades>=3 test, DD>-40%
Timeframe: 1d
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
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
    """Relative Strength Index - momentum oscillator"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] > 1e-10:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi

def calculate_crsi(close):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile of today's return vs last 100 days
    """
    n = len(close)
    if n < 100:
        return np.full(n, np.nan)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2)
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    streak[:] = np.nan
    streak_rsi[:] = np.nan
    
    current_streak = 0
    for i in range(1, n):
        if close[i] > close[i-1]:
            if current_streak > 0:
                current_streak += 1
            else:
                current_streak = 1
        elif close[i] < close[i-1]:
            if current_streak < 0:
                current_streak -= 1
            else:
                current_streak = -1
        else:
            current_streak = 0
        streak[i] = current_streak
    
    # Calculate RSI of streak
    for i in range(2, n):
        if np.isnan(streak[i]) or np.isnan(streak[i-1]):
            continue
        streak_changes = streak[max(0,i-1):i+1]
        if len(streak_changes) >= 2:
            delta_s = np.diff(streak_changes)
            gain_s = np.sum(np.where(delta_s > 0, delta_s, 0.0))
            loss_s = np.sum(np.where(delta_s < 0, -delta_s, 0.0))
            if loss_s > 1e-10:
                rs_s = gain_s / loss_s
                streak_rsi[i] = 100.0 - (100.0 / (1.0 + rs_s))
            else:
                streak_rsi[i] = 100.0
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    percent_rank[:] = np.nan
    returns = np.diff(close, prepend=close[0]) / (close + 1e-10)
    
    for i in range(100, n):
        window = returns[max(0,i-99):i+1]
        if len(window) >= 100:
            count_below = np.sum(window[:-1] < returns[i])
            percent_rank[i] = 100.0 * count_below / 99.0
    
    # Combine into CRSI
    crsi = np.zeros(n)
    crsi[:] = np.nan
    for i in range(100, n):
        if not np.isnan(rsi_3[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index = 100 * (ATR(1,sum of ATR(14))) / (Highest High - Lowest Low)
    
    CHOP > 61.8 = range/choppy market (mean-revert)
    CHOP < 38.2 = trending market (trend-follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Sum of ATR(14)
    atr_sum = np.zeros(n)
    atr_sum[:] = np.nan
    for i in range(period, n):
        atr_sum[i] = np.sum(tr[i-period+1:i+1])
    
    # Highest High - Lowest Low over period
    hh_ll = np.zeros(n)
    hh_ll[:] = np.nan
    for i in range(period, n):
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        hh_ll[i] = hh - ll
    
    # Choppiness Index
    chop = np.zeros(n)
    chop[:] = np.nan
    for i in range(period, n):
        if hh_ll[i] > 1e-10 and not np.isnan(atr_sum[i]):
            chop[i] = 100.0 * atr_sum[i] / hh_ll[i]
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMA
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1d indicators
    hma_1d = calculate_hma(close, period=21)
    sma_200 = calculate_sma(close, period=200)
    crsi = calculate_crsi(close)
    chop = calculate_choppiness(high, low, close, period=14)
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
    
    for i in range(250, n):  # Need 200 for SMA + 100 for CRSI + buffer
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(sma_200[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
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
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range between 55-62 for more trades
        is_trending = chop[i] < 45.0  # Range between 38-45 for more trades
        
        # === HTF BIAS (1w HMA) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === 1d TREND ===
        htf_1d_bull = close[i] > hma_1d[i]
        htf_1d_bear = close[i] < hma_1d[i]
        
        # === CRSI CONDITIONS (LOOSE for more trades) ===
        crsi_oversold = crsi[i] < 25.0  # Was 15, now 25 for more trades
        crsi_overbought = crsi[i] > 75.0  # Was 85, now 75 for more trades
        crsi_extreme_oversold = crsi[i] < 15.0
        crsi_extreme_overbought = crsi[i] > 85.0
        
        # === ENTRY LOGIC (DUAL REGIME) ===
        desired_signal = 0.0
        
        # LONG in CHOPPY regime (mean-reversion)
        if is_choppy:
            if crsi_oversold and close[i] > sma_200[i]:
                if crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # LONG in TRENDING regime (trend-follow with pullback)
        elif is_trending:
            if htf_1w_bull and htf_1d_bull and crsi_oversold:
                if crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT in CHOPPY regime (mean-reversion)
        if is_choppy:
            if crsi_overbought and close[i] < sma_200[i]:
                if crsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # SHORT in TRENDING regime (trend-follow with pullback)
        elif is_trending:
            if htf_1w_bear and htf_1d_bear and crsi_overbought:
                if crsi_extreme_overbought:
                    desired_signal = -SIZE_STRONG
                else:
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