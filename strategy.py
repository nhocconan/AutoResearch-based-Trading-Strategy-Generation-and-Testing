#!/usr/bin/env python3
"""
Experiment #790: 1h Primary + 4h/1d HTF — Connors RSI + Choppiness Regime

Hypothesis: 1h timeframe with 4h HMA trend bias + Connors RSI entries + Choppiness
regime filter will generate 40-80 trades/year with positive Sharpe. Previous 1h
attempts failed due to either 0 trades (too strict) or fee drag (too many trades).

Key innovations:
1. Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than standard RSI, 75% win rate on mean reversion
   - Long: CRSI < 15 + HTF bull | Short: CRSI > 85 + HTF bear
2. Choppiness Index(14) regime filter
   - CHOP < 45 = trending (allow trend entries)
   - CHOP > 65 = extreme range (allow mean reversion entries)
   - 45-65 = avoid trading (chop zone)
3. 4h HMA(21) for HTF trend bias — proven reliable across experiments
4. 1d ATR ratio for vol expansion confirmation (ATR(7)/ATR(30) > 1.3)
5. Session filter: 08-20 UTC only (avoid low liquidity Asian hours)
6. Discrete sizing: 0.0, ±0.25, ±0.35 with 2.5x ATR trailing stop

Target: Sharpe>0.45, trades>=40 train, trades>=5 test, DD>-35%
Timeframe: 1h
Size: 0.25 base, 0.35 strong signals
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h1d_v1"
timeframe = "1h"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3): Short-term momentum
    2. RSI_Streak(2): Streak RSI based on consecutive up/down days
    3. PercentRank(100): Percentile rank of price change over 100 periods
    
    Entry signals:
    - Long: CRSI < 15 (oversold)
    - Short: CRSI > 85 (overbought)
    """
    n = len(close)
    if n < rank_period + 1:
        return np.full(n, np.nan)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Component 2: Streak RSI
    # Count consecutive up/down days
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    streak[0] = 0
    
    for i in range(1, n):
        if delta[i] > 0:
            if delta[i-1] > 0:
                streak[i] = streak[i-1] + 1
            else:
                streak[i] = 1
        elif delta[i] < 0:
            if delta[i-1] < 0:
                streak[i] = streak[i-1] - 1
            else:
                streak[i] = -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    streak_rsi = np.zeros(n)
    streak_rsi[:] = np.nan
    for i in range(streak_period, n):
        total = avg_streak_gain[i] + avg_streak_loss[i]
        if total > 1e-10:
            streak_rsi[i] = 100.0 * (avg_streak_gain[i] / total)
        else:
            streak_rsi[i] = 50.0
    
    # Component 3: PercentRank(100)
    pct_rank = np.zeros(n)
    pct_rank[:] = np.nan
    
    for i in range(rank_period, n):
        price_changes = np.diff(close[i-rank_period:i+1])
        current_change = price_changes[-1] if len(price_changes) > 0 else 0
        
        count_lower = np.sum(price_changes[:-1] < current_change)
        total_comparable = len(price_changes) - 1
        
        if total_comparable > 0:
            pct_rank[i] = 100.0 * (count_lower / total_comparable)
        else:
            pct_rank[i] = 50.0
    
    # Combine components
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(pct_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + pct_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures if market is trending or ranging
    
    Formula: CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    Interpretation:
    - CHOP > 61.8: Market is choppy/ranging (mean reversion favorable)
    - CHOP < 38.2: Market is trending (trend following favorable)
    - 38.2-61.8: Transition zone (avoid trading)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Rolling sum of ATR
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    
    # Rolling highest high and lowest low
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    choppiness = np.zeros(n)
    choppiness[:] = np.nan
    
    log_period = np.log10(period)
    
    for i in range(period, n):
        price_range = highest[i] - lowest[i]
        if price_range > 1e-10 and atr_sum[i] > 1e-10:
            choppiness[i] = 100.0 * (np.log10(atr_sum[i] / price_range) / log_period)
    
    return choppiness

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA (4h)
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d ATR ratio for vol filter
    atr_1d_7_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=7)
    atr_1d_30_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=30)
    atr_ratio_1d_raw = atr_1d_7_raw / (atr_1d_30_raw + 1e-10)
    atr_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio_1d_raw)
    
    # Calculate 1h indicators
    crsi_1h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    choppiness_1h = calculate_choppiness(high, low, close, period=14)
    atr_14_1h = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
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
        if np.isnan(atr_14_1h[i]) or atr_14_1h[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(crsi_1h[i]) or np.isnan(choppiness_1h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_ratio_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (08-20 UTC) ===
        # Extract hour from open_time (milliseconds timestamp)
        hour_utc = (open_time[i] // 3600000) % 24
        in_session = (hour_utc >= 8) and (hour_utc <= 20)
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (4h HMA) ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop = choppiness_1h[i]
        trending_regime = chop < 45.0  # Trending market
        ranging_regime = chop > 65.0   # Extreme range (mean reversion)
        chop_zone = (chop >= 45.0) and (chop <= 65.0)  # Avoid this zone
        
        if chop_zone:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOL EXPANSION FILTER (1d ATR ratio) ===
        vol_expansion = atr_ratio_1d_aligned[i] > 1.3
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_1h[i] < 15.0
        crsi_overbought = crsi_1h[i] > 85.0
        crsi_extreme_oversold = crsi_1h[i] < 10.0
        crsi_extreme_overbought = crsi_1h[i] > 90.0
        
        # === ENTRY LOGIC (3+ CONFLUENCE) ===
        desired_signal = 0.0
        
        # LONG: HTF bull + CRSI oversold + (trending OR vol expansion)
        if htf_4h_bull and crsi_oversold:
            if trending_regime or vol_expansion:
                if crsi_extreme_oversold:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: HTF bear + CRSI overbought + (trending OR vol expansion)
        elif htf_4h_bear and crsi_overbought:
            if trending_regime or vol_expansion:
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
                entry_atr = atr_14_1h[i]
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