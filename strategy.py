#!/usr/bin/env python3
"""
Experiment #287: 1d Primary + 1w HTF — Regime-Adaptive CRSI/Donchian

Hypothesis: BTC/ETH 2025+ is bear/range market (not bull like 2021-2024).
Simple trend following fails. Need regime-adaptive logic:
- CHOP > 61.8 = range market → Connors RSI mean reversion
- CHOP < 38.2 = trending market → Donchian breakout + HMA trend
- 1w HMA for macro bias (only short if price < 1w HMA, only long if price > 1w HMA)

Why 1d: Proven to work (#277 Sharpe=0.294, #283 Sharpe=0.267). Natural 20-50 trades/year.
Why CRSI: 75% win rate on mean reversion, catches bear market reversals.
Why Donchian: Catches true breakouts in trending regimes.

TARGET: Sharpe > 0.5 on ALL symbols, 20-50 trades/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_crsi_donchian_1w_hma_atr_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    tr_s = pd.Series(tr)
    
    atr_sum = tr_s.rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    return np.nan_to_num(chop, nan=50.0)

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak RSI: consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) of streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.clip(lower=0)
    streak_loss = (-streak_delta).clip(lower=0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
        rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.fillna(50.0).values
    
    # Percent Rank (100): where current close ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # CRSI = average of 3 components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    return crsi

def calculate_donchian_channels(high, low, period=20):
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, period=20)
    
    # Calculate and align 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Conservative for daily volatility
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):  # Need warmup for CRSI (100) + Donchian (20) + ATR (14)
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            signals[i] = 0.0
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 55.0  # Range market (slightly lower threshold for more trades)
        is_trending = chop_14[i] < 45.0  # Trending market
        
        # === MACRO BIAS (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_aligned[i]
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # --- MEAN REVERSION MODE (choppy market) ---
        if is_choppy:
            # Long: CRSI < 15 (oversold) + price above 1w HMA (macro bullish)
            if crsi[i] < 15.0 and price_above_1w_hma:
                desired_signal = POSITION_SIZE
            # Short: CRSI > 85 (overbought) + price below 1w HMA (macro bearish)
            elif crsi[i] > 85.0 and price_below_1w_hma:
                desired_signal = -POSITION_SIZE
        
        # --- TREND FOLLOWING MODE (trending market) ---
        elif is_trending:
            # Long: Break above Donchian upper + price above 1w HMA
            if close[i] > donchian_upper[i] and price_above_1w_hma:
                desired_signal = POSITION_SIZE
            # Short: Break below Donchian lower + price below 1w HMA
            elif close[i] < donchian_lower[i] and price_below_1w_hma:
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
        
        # === CRSI EXIT (take profit in mean reversion mode) ===
        if in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC ===
        # If already in position and no stoploss/exit, maintain position
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if not overbought
                if crsi[i] < 80.0:
                    desired_signal = POSITION_SIZE
            elif position_side < 0:
                # Hold short if not oversold
                if crsi[i] > 20.0:
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
                # Position flip
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