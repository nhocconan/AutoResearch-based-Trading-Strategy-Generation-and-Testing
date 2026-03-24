#!/usr/bin/env python3
"""
Experiment #936: 30m Primary + 4h/1d HTF — Regime-Adaptive CRSI + Choppiness + HMA

Hypothesis: 30m timeframe with 4h/1d HTF bias provides optimal trade frequency (40-80/year)
while maintaining signal quality. Key innovation: regime-adaptive logic using Choppiness
Index to switch between trend-following (CHOP<38.2) and mean-reversion (CHOP>61.8).
Connors RSI (CRSI) captures short-term extremes for precise entry timing.

Key innovations:
1. 1d HMA(21) for primary trend bias - price above = bullish, below = bearish
2. 4h Choppiness Index(14) for regime detection - trend vs range
3. 30m CRSI(3,2,100) for entry timing - extreme oversold/overbought
4. Session filter (08-20 UTC) to reduce trades and avoid low-liquidity periods
5. ATR(14) 2.5x trailing stop for risk management
6. Discrete sizing: 0.0, ±0.20, ±0.25 to minimize fee churn

Regime-adaptive entry logic:
- TREND REGIME (CHOP < 38.2): Follow HTF trend, enter on CRSI pullback (CRSI<30 long, >70 short)
- RANGE REGIME (CHOP > 61.8): Mean revert, enter on CRSI extremes (CRSI<15 long, >85 short)
- NEUTRAL (38.2 <= CHOP <= 61.8): No trades, wait for clarity

Session filter: Only trade 08:00-20:00 UTC (high liquidity, lower spread)

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-40%
Timeframe: 30m
Size: 0.20-0.25 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_crsi_chop_hma_4h1d_v1"
timeframe = "30m"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI)
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI applied to streak of consecutive up/down days
    PercentRank: Percentile rank of price change over lookback period
    """
    n = len(close)
    if n < rank_period + 5:
        return np.full(n, np.nan)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    rsi_streak = calculate_rsi(streak, streak_period)
    
    # PercentRank(100) - percentile rank of price change
    pct_rank = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        changes = np.diff(close[i-rank_period:i+1])
        if len(changes) > 0:
            current_change = close[i] - close[i-1]
            rank = np.sum(changes < current_change) / len(changes)
            pct_rank[i] = rank * 100.0
    
    # Combine into CRSI
    crsi = np.full(n, np.nan, dtype=np.float64)
    for i in range(rank_period, n):
        if not np.isnan(rsi_close[i]) and not np.isnan(rsi_streak[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_close[i] + rsi_streak[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    
    CHOP > 61.8 = range/consolidation (mean revert)
    CHOP < 38.2 = trend (trend follow)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    choppiness = np.full(n, np.nan, dtype=np.float64)
    log_n = np.log10(period)
    
    for i in range(period, n):
        if highest[i] > lowest[i] and atr_sum[i] > 0:
            choppiness[i] = 100.0 * np.log10(atr_sum[i] / (highest[i] - lowest[i])) / log_n
    
    return choppiness

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

def calculate_sma(close, period):
    """Simple Moving Average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def is_session_active(open_time):
    """
    Session filter: Only trade 08:00-20:00 UTC
    open_time is in milliseconds since epoch
    """
    # Convert to hour of day UTC
    hour = (open_time // 3600000) % 24
    return 8 <= hour < 20

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 30m indicators
    hma_30m_16 = calculate_hma(close, period=16)
    hma_30m_48 = calculate_hma(close, period=48)
    crsi_30m = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(250, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_30m_16[i]) or np.isnan(hma_30m_48[i]) or np.isnan(crsi_30m[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER ===
        if not is_session_active(open_time[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1d HMA) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        chop = chop_4h_aligned[i]
        trend_regime = chop < 38.2
        range_regime = chop > 61.8
        neutral_regime = 38.2 <= chop <= 61.8
        
        # === 30m HMA TREND ===
        hma_30m_bull = hma_30m_16[i] > hma_30m_48[i]
        hma_30m_bear = hma_30m_16[i] < hma_30m_48[i]
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else False
        price_below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === CRSI EXTREMES ===
        crsi_oversold = crsi_30m[i] < 15
        crsi_very_oversold = crsi_30m[i] < 10
        crsi_overbought = crsi_30m[i] > 85
        crsi_very_overbought = crsi_30m[i] > 90
        
        # CRSI pullback levels for trend regime
        crsi_pullback_long = crsi_30m[i] < 30
        crsi_pullback_short = crsi_30m[i] > 70
        
        # === ENTRY LOGIC (REGIME-ADAPTIVE) ===
        desired_signal = 0.0
        
        # TREND REGIME: Follow HTF trend, enter on pullback
        if trend_regime:
            if htf_1d_bull and hma_30m_bull and price_above_sma200 and crsi_pullback_long:
                desired_signal = SIZE_BASE
            elif htf_1d_bull and hma_30m_bull and price_above_sma200 and crsi_very_oversold:
                desired_signal = SIZE_STRONG
            
            if htf_1d_bear and hma_30m_bear and price_below_sma200 and crsi_pullback_short:
                desired_signal = -SIZE_BASE
            elif htf_1d_bear and hma_30m_bear and price_below_sma200 and crsi_very_overbought:
                desired_signal = -SIZE_STRONG
        
        # RANGE REGIME: Mean revert at extremes
        elif range_regime:
            if htf_1d_bull and crsi_very_oversold and price_above_sma200:
                desired_signal = SIZE_BASE
            elif htf_1d_bear and crsi_very_overbought and price_below_sma200:
                desired_signal = -SIZE_BASE
        
        # NEUTRAL REGIME: No trades (wait for clarity)
        # desired_signal stays 0.0
        
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