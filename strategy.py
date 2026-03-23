#!/usr/bin/env python3
"""
Experiment #326: 12h Primary + 1d HTF — Dual Regime with Fisher + CRSI + Volume

Hypothesis: 12h timeframe reduces fee drag while maintaining edge. Key improvements:
1. Choppiness Index for regime detection (range vs trend)
2. Connors RSI + Fisher Transform for mean reversion in range (dual confirmation)
3. Donchian breakout for trend following
4. Volume confirmation on breakouts (reduces false signals)
5. 1d HMA for directional bias (NOT hard filter - allows counter-trend trades)
6. LOOSE thresholds to ensure 30+ trades/year (learned from #318, #319 failures)

Key difference from #321: Added Fisher Transform for reversal confirmation + Volume filter
on breakouts to improve win rate. 12h naturally has fewer trades = less fee drag.

TARGET: 25-40 trades/year, Sharpe > 0.7 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_regime_fisher_crsi_donchian_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
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
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_s = pd.Series(streak_abs)
    gain_streak = streak_s.diff().clip(lower=0)
    loss_streak = (-streak_s.diff()).clip(lower=0)
    avg_gain_streak = gain_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_loss_streak = loss_streak.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_gain_streak / (avg_loss_streak + 1e-10)
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

def calculate_fisher(close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to -1 to +1, then applies Fisher transform.
    Good for catching reversals in bear/range markets.
    """
    close_s = pd.Series(close)
    
    highest = close_s.rolling(window=period, min_periods=period).max()
    lowest = close_s.rolling(window=period, min_periods=period).min()
    
    normalized = 2 * ((close_s - lowest) / (highest - lowest + 1e-10)) - 1
    normalized = np.clip(normalized, -0.999, 0.999)
    
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    fisher = fisher.fillna(0)
    
    return fisher.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range, CHOP < 38.2 = trend
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / (vol_avg + 1e-10)
    return vol_ratio.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    fisher = calculate_fisher(close, period=9)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Calculate and align 1d HMA for macro bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28
    
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
        if np.isnan(rsi_14[i]) or np.isnan(crsi[i]) or np.isnan(fisher[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (1d HMA) — DIRECTIONAL BIAS ONLY ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # Thresholds: >55 = choppy/range, <45 = trending
        is_choppy = chop[i] > 55.0
        is_trending = chop[i] < 45.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: CRSI + Fisher Mean Reversion
            # LOOSE thresholds: CRSI<25 long, >75 short (ensures trades)
            # Fisher confirmation: fisher < -0.8 for long, > 0.8 for short
            if crsi[i] < 25.0 and fisher[i] < -0.8:
                if price_above_hma_1d:
                    desired_signal = POSITION_SIZE  # Bull + oversold = strong long
                else:
                    desired_signal = POSITION_SIZE * 0.5  # Bear + oversold = weak long
            elif crsi[i] > 75.0 and fisher[i] > 0.8:
                if price_below_hma_1d:
                    desired_signal = -POSITION_SIZE  # Bear + overbought = strong short
                else:
                    desired_signal = -POSITION_SIZE * 0.5  # Bull + overbought = weak short
        
        else:  # is_trending or neutral (45-55)
            # TREND REGIME: Donchian Breakout + Volume confirmation
            # Volume filter: only take breakouts with vol_ratio > 1.2
            vol_confirmed = vol_ratio[i] > 1.2
            
            if close[i] > donchian_upper[i-1]:
                if vol_confirmed or price_above_hma_1d:
                    if price_above_hma_1d:
                        desired_signal = POSITION_SIZE
                    else:
                        desired_signal = POSITION_SIZE * 0.5
            elif close[i] < donchian_lower[i-1]:
                if vol_confirmed or price_below_hma_1d:
                    if price_below_hma_1d:
                        desired_signal = -POSITION_SIZE
                    else:
                        desired_signal = -POSITION_SIZE * 0.5
        
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
        
        # === CRSI EXIT (take profit in range regime) ===
        if is_choppy and in_position and position_side > 0 and crsi[i] > 65.0:
            desired_signal = 0.0
        
        if is_choppy and in_position and position_side < 0 and crsi[i] < 35.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                desired_signal = POSITION_SIZE
            elif position_side < 0:
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