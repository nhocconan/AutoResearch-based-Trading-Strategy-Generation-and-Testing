#!/usr/bin/env python3
"""
Experiment #273: 1d Primary + 1w HTF — Dual Regime Strategy with Connors RSI

Hypothesis: Daily timeframe with weekly trend filter + regime detection can beat the current best (Sharpe=0.486).

Key components:
- 1w HMA(21) for MACRO trend direction (slow, reliable)
- 1d Choppiness Index(14) for regime detection (range vs trend)
- Connors RSI for entry timing (mean reversion in range, trend follow otherwise)
- ATR(14) 2.5x trailing stoploss
- Position size: 0.28 (conservative for daily volatility)

Why this might work:
1. 1d timeframe has proven successful in past experiments
2. Connors RSI has 75% win rate for mean reversion
3. Choppiness Index filters out bad trend entries during ranges
4. Weekly HMA provides stable macro bias
5. Dual regime adapts to market conditions

TARGET: 20-40 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_regime_1w_hma_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, period)
    hull = 2 * wma_half - wma_full
    hma = wma(hull, sqrt_n)
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """Calculate Connors RSI (CRSI)."""
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    rsi_3 = rsi_3.fillna(50.0).values
    
    # Component 2: RSI of Streak (2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + streak_rs))
    streak_rsi = streak_rsi.fillna(50.0).values
    
    # Component 3: Percent Rank (100)
    percent_rank = close_s.rolling(window=pr_period, min_periods=pr_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100 if x.max() > x.min() else 50.0,
        raw=False
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    # Connors RSI = average of three components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Sum of TR over period
    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high - lowest low over period
    hh_ll = high_s.rolling(window=period, min_periods=period).max() - \
            low_s.rolling(window=period, min_periods=period).min()
    
    # Choppiness formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_tr / hh_ll) / np.log10(period)
    
    chop = chop.fillna(50.0).values
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load weekly HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Calculate and align 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for daily
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1w HMA) ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 61.8 = ranging market (mean reversion)
        # CHOP < 38.2 = trending market (trend follow)
        is_ranging = chop[i] > 55.0  # Slightly relaxed threshold
        is_trending = chop[i] < 45.0  # Slightly relaxed threshold
        
        # === CONNORS RSI SIGNALS ===
        # Long: CRSI < 15 (oversold)
        # Short: CRSI > 85 (overbought)
        crsi_oversold = crsi[i] < 15.0
        crsi_overbought = crsi[i] > 85.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG ENTRY
        if price_above_hma_1w:  # Macro bullish
            if is_ranging and crsi_oversold:
                # Mean reversion in range
                desired_signal = POSITION_SIZE
            elif is_trending and crsi_oversold:
                # Trend pullback
                desired_signal = POSITION_SIZE
        
        # SHORT ENTRY
        if price_below_hma_1w:  # Macro bearish
            if is_ranging and crsi_overbought:
                # Mean reversion in range
                desired_signal = -POSITION_SIZE
            elif is_trending and crsi_overbought:
                # Trend pullback
                desired_signal = -POSITION_SIZE
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === REGIME CHANGE EXIT ===
        # Exit if regime changes against position
        if in_position and position_side > 0 and chop[i] > 65.0:
            # Long in newly choppy market - reduce risk
            if crsi[i] > 50.0:  # And CRSI recovered
                desired_signal = 0.0
        
        if in_position and position_side < 0 and chop[i] > 65.0:
            # Short in newly choppy market - reduce risk
            if crsi[i] < 50.0:  # And CRSI recovered
                desired_signal = 0.0
        
        # === CRSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 80.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 20.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_hma_1w:
                desired_signal = POSITION_SIZE
            elif position_side < 0 and price_below_hma_1w:
                desired_signal = -POSITION_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else float('inf')
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals