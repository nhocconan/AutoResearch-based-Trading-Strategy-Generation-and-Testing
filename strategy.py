#!/usr/bin/env python3
"""
Experiment #284: 4h Primary + 12h/1d HTF — Adaptive Regime Strategy

Hypothesis: Previous 4h strategies failed from being either too trend-focused (#279 Sharpe=-0.612)
or too chop-focused (#282 Sharpe=-0.042). This version ADAPTS to regime:
- Choppiness Index(14) > 61.8 = RANGE regime → use CRSI mean reversion
- Choppiness Index(14) < 38.2 = TREND regime → use HMA + Donchian breakout
- Between 38.2-61.8 = neutral → reduce position size by 50%

KEY INSIGHTS from failures:
- #278/#280 got 0 trades from over-filtering (session + too many confluence)
- #279 negative Sharpe from pure trend (failed in 2022 crash + 2025 bear)
- Need BOTH mean reversion AND trend following, switched by regime

HTF Usage:
- 12h HMA(21) for macro bias (soft filter, not hard requirement)
- 1d HMA(50) for secular trend (only affects position sizing)

Position sizing: 0.30 normal, 0.15 in neutral regime, 0.0 in extreme chop (>75)
Stoploss: 2.5x ATR trailing

TARGET: 25-45 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_regime_adaptive_crsi_donchian_12h1d_atr_v1"
timeframe = "4h"
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
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) on close
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_close = 100.0 - (100.0 / (1.0 + rs))
    rsi_close = rsi_close.fillna(50.0)
    
    # RSI on streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.clip(lower=0)
    streak_loss = (-streak_s).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0)
    
    # PercentRank(100) - where current close ranks vs last 100 closes
    percent_rank = pd.Series(close).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    High CHOP (>61.8) = ranging market
    Low CHOP (<38.2) = trending market
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    tr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(tr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0, posinf=50.0, neginf=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper/lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_16 = calculate_hma(close, 16)
    hma_48 = calculate_hma(close, 48)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi_3 = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 12h HMA for medium-term bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for long-term bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    
    # Position sizing levels (discrete to minimize fee churn)
    SIZE_TREND = 0.30      # Full size in clear trend
    SIZE_CHOP = 0.25       # Slightly smaller in chop (mean reversion)
    SIZE_NEUTRAL = 0.15    # Half size in neutral regime
    
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
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            continue
        if np.isnan(crsi_3[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop = chop_14[i]
        
        if chop > 61.8:
            regime = 'CHOP'      # Mean reversion
            position_size = SIZE_CHOP
        elif chop < 38.2:
            regime = 'TREND'     # Trend following
            position_size = SIZE_TREND
        else:
            regime = 'NEUTRAL'   # Reduced size
            position_size = SIZE_NEUTRAL
        
        # Skip extreme chop (>75) - too dangerous
        if chop > 75.0:
            signals[i] = 0.0
            continue
        
        # === MACRO BIAS (12h/1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_bullish = hma_16[i] > hma_48[i]
        hma_bearish = hma_16[i] < hma_48[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if regime == 'CHOP':
            # MEAN REVERSION: CRSI extremes
            # Long when CRSI < 15 (oversold) + price above 12h HMA (soft bias)
            if crsi_3[i] < 15.0 and price_above_hma_12h:
                desired_signal = position_size
            # Short when CRSI > 85 (overbought) + price below 12h HMA (soft bias)
            elif crsi_3[i] > 85.0 and not price_above_hma_12h:
                desired_signal = -position_size
        
        elif regime == 'TREND':
            # TREND FOLLOWING: HMA + Donchian breakout
            # Long when HMA bullish + price breaks Donchian upper
            if hma_bullish and close[i] > donchian_upper[i] and price_above_hma_12h:
                desired_signal = position_size
            # Short when HMA bearish + price breaks Donchian lower
            elif hma_bearish and close[i] < donchian_lower[i] and not price_above_hma_12h:
                desired_signal = -position_size
        
        else:  # NEUTRAL
            # Only enter on extreme signals
            if crsi_3[i] < 10.0 and price_above_hma_12h:
                desired_signal = position_size
            elif crsi_3[i] > 90.0 and not price_above_hma_12h:
                desired_signal = -position_size
        
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
        
        # === TREND REVERSAL EXIT ===
        if in_position and position_side > 0 and hma_bearish and regime == 'TREND':
            desired_signal = 0.0
        
        if in_position and position_side < 0 and hma_bullish and regime == 'TREND':
            desired_signal = 0.0
        
        # === CRSI MEAN REVERSION EXIT (take profit) ===
        if in_position and position_side > 0 and crsi_3[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi_3[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC (maintain position if still valid) ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if regime still supports or CRSI not overbought
                if (regime == 'CHOP' and crsi_3[i] < 70.0) or (regime == 'TREND' and hma_bullish):
                    desired_signal = position_size
            elif position_side < 0:
                # Hold short if regime still supports or CRSI not oversold
                if (regime == 'CHOP' and crsi_3[i] > 30.0) or (regime == 'TREND' and hma_bearish):
                    desired_signal = -position_size
        
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