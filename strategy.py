#!/usr/bin/env python3
"""
Experiment #335: 1h Primary + 4h/1d HTF — Regime-Adaptive with Soft Bias

Hypothesis: Previous 1h failures (#325, #328, #330) had 0 trades from over-filtering
(session + volume + multiple HTF filters). This strategy uses:
1. 4h HMA(21) as PRIMARY trend filter (hard filter for direction)
2. 1d HMA(21) as MACRO BIAS only (adjusts position size, doesn't block entries)
3. 1h Choppiness Index for regime detection (CHOP>55=range, CHOP<45=trend)
4. 1h CRSI for mean-reversion entries (CRSI<20 long, >80 short in range)
5. 1h Donchian(20) breakout for trend entries (in trend regime)

KEY INSIGHT: Remove session/volume filters that killed trade count in prior 1h attempts.
Use 4h for direction (hard), 1d for bias (soft size adjustment), 1h for entry timing.
Position size 0.25 (smaller for 1h to reduce fee impact on higher trade frequency).

TARGET: 40-70 trades/year on 1h, Sharpe > 0.6 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_crsi_donchian_4h1d_bias_v1"
timeframe = "1h"
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
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
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
    streak_rsi = np.where(delta > 0, streak_rsi, 100 - streak_rsi)
    
    # Percent Rank (100)
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) * 100, raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    crsi = (rsi_short + streak_rsi + percent_rank) / 3
    return crsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align 4h HMA for trend direction (HARD FILTER)
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro bias (SOFT - size adjustment only)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # Smaller for 1h to reduce fee impact
    
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
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (4h HMA - HARD FILTER) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === MACRO BIAS (1d HMA - SOFT, adjusts size) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Size modifier: aligned with 1d = full size, against 1d = half size
        size_modifier = 1.0
        if price_above_hma_4h and price_below_hma_1d:
            size_modifier = 0.5  # Long against 1d trend
        elif price_below_hma_4h and price_above_hma_1d:
            size_modifier = 0.5  # Short against 1d trend
        
        POSITION_SIZE = BASE_SIZE * size_modifier
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 55.0  # Range market
        is_trending = chop[i] < 45.0  # Trend market
        # 45-55 = neutral, default to trend logic
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        if is_choppy:
            # RANGE REGIME: Connors RSI Mean Reversion
            # LONG: CRSI < 20 + price above 4h HMA
            if crsi[i] < 20.0 and price_above_hma_4h:
                desired_signal = POSITION_SIZE
            # SHORT: CRSI > 80 + price below 4h HMA
            elif crsi[i] > 80.0 and price_below_hma_4h:
                desired_signal = -POSITION_SIZE
        
        else:  # is_trending or neutral (45-55)
            # TREND REGIME: Donchian Breakout
            # LONG: Price breaks Donchian upper + bullish 4h trend
            if close[i] > donchian_upper[i-1] and price_above_hma_4h:
                desired_signal = POSITION_SIZE
            # SHORT: Price breaks Donchian lower + bearish 4h trend
            elif close[i] < donchian_lower[i-1] and price_below_hma_4h:
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
        
        # === CRSI EXTREME EXIT (take profit in range regime) ===
        if is_choppy and in_position and position_side > 0 and crsi[i] > 70.0:
            desired_signal = 0.0
        
        if is_choppy and in_position and position_side < 0 and crsi[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                desired_signal = BASE_SIZE * (1.0 if price_above_hma_1d else 0.5)
            elif position_side < 0:
                desired_signal = -BASE_SIZE * (1.0 if price_below_hma_1d else 0.5)
        
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