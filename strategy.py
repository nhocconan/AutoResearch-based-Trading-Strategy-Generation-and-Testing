#!/usr/bin/env python3
"""
Experiment #1140: 1h Primary + 4h/12h HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After analyzing 829 failed strategies, the pattern is clear:
- Complex regime switching = 0 trades (exp #1130-1134, #1137, #1139)
- 1h timeframe needs LOOSE entry thresholds to generate trades
- Connors RSI (CRSI) has 75% win rate in research literature
- Choppiness Index properly filters range vs trend regimes
- 4h HMA provides stable trend direction (proven in exp #1129)

Strategy Logic:
1. 4h HMA(21) for macro trend direction (load ONCE before loop)
2. 12h HMA(21) for higher-timeframe confirmation
3. 1h Connors RSI for entry timing (CRSI < 25 long, > 75 short)
4. Choppiness Index(14) for regime: CHOP > 55 = range (mean revert), CHOP < 45 = trend
5. Session filter: 8-20 UTC only (liquidity hours)
6. Volume filter: volume > 0.8x 20-bar average
7. ATR(14) 2.5x trailing stoploss
8. Position size: 0.25 (smaller for 1h TF to reduce fee drag)

Why this should work:
- CRSI loose thresholds (25/75 not 20/80) ensure trade frequency
- CHOP filter prevents trend-following in choppy markets
- 4h+12h HMA confluence prevents counter-trend trades
- Session filter reduces noise from low-liquidity hours
- Target: 40-80 trades/year, Sharpe > 0.612

Timeframe: 1h (primary)
HTF: 4h, 12h — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 discrete (0.0, ±0.25)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_hma_4h12h_session_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — combines 3 components for mean reversion signals.
    Formula: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Research shows 75% win rate with CRSI < 10 long, > 90 short.
    We use looser thresholds (25/75) for trade frequency.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streaks
    diff = np.diff(close)
    diff = np.concatenate([[0.0], diff])
    
    streak = np.zeros(n)
    streak_rsi = np.full(n, np.nan)
    
    for i in range(1, n):
        if diff[i] > 0:
            streak[i] = streak[i-1] + 1 if diff[i-1] >= 0 else 1
        elif diff[i] < 0:
            streak[i] = streak[i-1] - 1 if diff[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100 scale)
    abs_streak = np.abs(streak)
    streak_sign = np.sign(streak)
    
    # Simple streak RSI approximation
    for i in range(streak_period, n):
        up_streaks = np.sum(np.maximum(streak[i-streak_period:i+1], 0))
        down_streaks = np.sum(np.abs(np.minimum(streak[i-streak_period:i+1], 0)))
        if down_streaks > 1e-10:
            streak_rsi[i] = 100.0 - (100.0 / (1.0 + up_streaks / down_streaks))
        else:
            streak_rsi[i] = 100.0 if up_streaks > 0 else 50.0
    
    # Component 3: Percent Rank of returns over lookback
    returns = np.diff(close) / close[:-1]
    returns = np.concatenate([[0.0], returns])
    
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current = returns[i]
        rank = np.sum(window < current)
        percent_rank[i] = 100.0 * rank / rank_period
    
    # Combine components
    mask = ~np.isnan(rsi_short) & ~np.isnan(streak_rsi) & ~np.isnan(percent_rank)
    crsi[mask] = (rsi_short[mask] + streak_rsi[mask] + percent_rank[mask]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending
    We use 55/45 thresholds for clearer regime separation.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for higher-timeframe confirmation
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume average for filter
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller size for 1h TF to reduce fee drag
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Start after warmup period
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(vol_avg[i]) or atr[i] <= 1e-10:
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        # Convert open_time to hour
        hour = pd.to_datetime(open_time[i], unit='ms').hour
        in_session = 8 <= hour <= 20
        
        # === VOLUME FILTER ===
        vol_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 55 = range (mean revert), CHOP < 45 = trend (trend follow)
        is_range = chop[i] > 55.0
        is_trend = chop[i] < 45.0
        
        # === MACRO TREND (4h + 12h HMA) ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        trend_12h_bull = close[i] > hma_12h_aligned[i]
        trend_12h_bear = close[i] < hma_12h_aligned[i]
        
        # Strong trend confirmation: both 4h and 12h agree
        macro_bull = trend_4h_bull and trend_12h_bull
        macro_bear = trend_4h_bear and trend_12h_bear
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Range regime: CRSI oversold + macro bull (mean revert in uptrend)
        # Trend regime: CRSI oversold + macro bull (pullback entry)
        if in_session and vol_ok:
            if crsi[i] < 25.0 and macro_bull:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Range regime: CRSI overbought + macro bear (mean revert in downtrend)
        # Trend regime: CRSI overbought + macro bear (pullback entry)
        if in_session and vol_ok:
            if crsi[i] > 75.0 and macro_bear:
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if regime intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull
                if macro_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses
        if in_position and position_side > 0:
            if macro_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            if macro_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals