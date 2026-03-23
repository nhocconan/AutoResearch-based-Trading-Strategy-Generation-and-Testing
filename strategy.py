#!/usr/bin/env python3
"""
Experiment #316: 12h Primary + 1d HTF — Adaptive Regime Strategy

Hypothesis: 12h timeframe provides optimal balance of signal quality (20-50 trades/year).
Combining Choppiness Index regime detection with ADX trend confirmation reduces false breakouts.
Using 1d HMA(21) for macro bias ensures we only trade with the higher timeframe trend.

Strategy Logic:
- REGIME: CHOP(14) + ADX(14)
  * CHOP > 55 = Range → CRSI mean reversion (long CRSI<15, short CRSI>85)
  * CHOP < 45 + ADX > 25 = Trend → Donchian breakout (long upper break, short lower break)
  
- MACRO FILTER: Price must be above 1d HMA for longs, below for shorts
- STOPLOSS: 2.5x ATR trailing stop
- EXIT: Regime change or 1d HMA cross against position
- Position Size: 0.30 (30% of capital, discrete levels)

TARGET: Sharpe > 0.612 (beat current best), 25-45 trades/year, DD < -35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_adaptive_crsi_donchian_1d_atr_v1"
timeframe = "12h"
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_short = 100.0 - (100.0 / (1.0 + rs))
    rsi_short = rsi_short.fillna(50.0)
    
    # RSI Streak (2)
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.diff().clip(lower=0)
    streak_loss = (-streak_s.diff()).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100.0 - (100.0 / (1.0 + rs_streak))
    streak_rsi = streak_rsi.fillna(50.0)
    streak_rsi = np.where(delta.values > 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank (100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    return crsi.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (tr_smooth + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean() / (tr_smooth + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum()
    
    chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    chop = chop.fillna(50.0).clip(0, 100)
    
    return chop.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 1d HMA
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.30
    
    # Position tracking
    position_side = 0
    entry_price = 0.0
    extreme_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(adx_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Macro bias
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Regime detection
        is_choppy = chop[i] > 55.0
        is_trending = (chop[i] < 45.0) and (adx_14[i] > 25.0)
        
        # Update trailing stop extremes if in position
        if position_side > 0:
            extreme_price = max(extreme_price, close[i])
        elif position_side < 0:
            extreme_price = min(extreme_price, close[i])
        
        # Check stoploss FIRST
        stoploss_triggered = False
        if position_side > 0:
            stop_price = extreme_price - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        elif position_side < 0:
            stop_price = extreme_price + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # Check regime change exit
        if i > 100:
            prev_chop = chop[i-1]
            prev_adx = adx_14[i-1]
            was_trending = (prev_chop < 45.0) and (prev_adx > 25.0)
            
            if position_side > 0 and is_choppy and was_trending:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                extreme_price = 0.0
                continue
            elif position_side < 0 and is_choppy and was_trending:
                signals[i] = 0.0
                position_side = 0
                entry_price = 0.0
                extreme_price = 0.0
                continue
        
        # Check 1d HMA cross exit
        if position_side > 0 and price_below_hma_1d:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        if position_side < 0 and price_above_hma_1d:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            extreme_price = 0.0
            continue
        
        # Determine desired signal
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: CRSI Mean Reversion
            if crsi[i] < 15.0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif crsi[i] > 85.0 and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        elif is_trending:
            # TREND REGIME: Donchian Breakout
            if close[i] > donchian_upper[i-1] and rsi_14[i] > 50.0 and price_above_hma_1d:
                desired_signal = POSITION_SIZE
            elif close[i] < donchian_lower[i-1] and rsi_14[i] < 50.0 and price_below_hma_1d:
                desired_signal = -POSITION_SIZE
        
        # Hold current position if no new signal but still valid
        if desired_signal == 0.0 and position_side != 0:
            desired_signal = position_side * POSITION_SIZE
        
        # Update position state
        if desired_signal > 0 and position_side != 1:
            position_side = 1
            entry_price = close[i]
            extreme_price = close[i]
        elif desired_signal < 0 and position_side != -1:
            position_side = -1
            entry_price = close[i]
            extreme_price = close[i]
        elif desired_signal == 0.0 and position_side != 0:
            position_side = 0
            entry_price = 0.0
            extreme_price = 0.0
        
        signals[i] = desired_signal
    
    return signals