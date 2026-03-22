#!/usr/bin/env python3
"""
Experiment #591: 1h Regime-Adaptive Strategy with 4h HMA + Connors RSI + Choppiness Index

Hypothesis: After 590+ failed experiments, the key insight is:
1. 1h timeframe needs REGIME DETECTION to avoid whipsaw (most 1h strategies fail here)
2. Choppiness Index (CHOP) distinguishes trend vs range markets reliably
3. Connors RSI (CRSI) has 75% win rate for mean reversion in range markets
4. 4h HMA provides trend bias without blocking all trades (unlike daily HMA)
5. Dual-mode: trend-follow in low CHOP, mean-revert in high CHOP
6. This should work in both 2021-2024 (trending) and 2025+ (bear/range)

Why this should beat #587 (Sharpe=-0.190):
- Regime detection prevents trend strategies in choppy markets
- Connors RSI catches reversals that pure trend misses
- 4h HMA is more responsive than daily HMA for 1h entries
- Adaptive position sizing reduces exposure in uncertain regimes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.35 discrete (max 0.40)
Stoploss: 2.5 * ATR(14) trailing (slightly wider for 1h noise)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_chop_4h_hma_connors_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Calculate ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Sum of ATR over period
    atr_sum = atr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    
    # Price range
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    Long when CRSI < 10, Short when CRSI > 90
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(close, 3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi_close = 100 - (100 / (1 + rs))
    rsi_close = rsi_close.fillna(50.0)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / avg_streak_loss
        rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50.0)
    
    # Percent Rank
    percent_rank = close_s.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50.0,
        raw=False
    )
    percent_rank = percent_rank.fillna(50.0)
    
    # CRSI
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    
    return crsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Additional trend filter
    close_s = pd.Series(close)
    ema_21 = close_s.ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_TREND = 0.30
    SIZE_MR = 0.25
    
    # Track position state for stoploss (separate from signal)
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(ema_50[i]):
            continue
        
        # === REGIME DETECTION ===
        trend_regime = chop_14[i] < 45.0  # Trending market
        range_regime = chop_14[i] > 55.0  # Ranging market
        # Neutral zone (45-55): use trend bias only
        
        # === 4H HMA TREND BIAS ===
        bull_bias = close[i] > hma_4h_aligned[i]
        bear_bias = close[i] < hma_4h_aligned[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # TREND REGIME: Follow 4h HMA direction with pullback
        if trend_regime:
            # Long: 4h bullish + price above EMA21 + pullback to EMA21
            if bull_bias and close[i] > ema_21[i] and close[i] < ema_21[i] * 1.005:
                new_signal = SIZE_TREND
            # Short: 4h bearish + price below EMA21 + pullback to EMA21
            elif bear_bias and close[i] < ema_21[i] and close[i] > ema_21[i] * 0.995:
                new_signal = -SIZE_TREND
        
        # RANGE REGIME: Connors RSI mean reversion
        elif range_regime:
            # Long: CRSI < 15 (oversold) + 4h bullish bias preferred
            if crsi[i] < 15 and bull_bias:
                new_signal = SIZE_MR
            elif crsi[i] < 10:  # Extreme oversold, enter anyway
                new_signal = SIZE_MR
            # Short: CRSI > 85 (overbought) + 4h bearish bias preferred
            elif crsi[i] > 85 and bear_bias:
                new_signal = -SIZE_MR
            elif crsi[i] > 90:  # Extreme overbought, enter anyway
                new_signal = -SIZE_MR
        
        # NEUTRAL REGIME: Only enter on strong 4h bias + CRSI extreme
        else:
            if bull_bias and crsi[i] < 12:
                new_signal = SIZE_MR * 0.7
            elif bear_bias and crsi[i] > 88:
                new_signal = -SIZE_MR * 0.7
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and bear_bias and chop_14[i] < 45:
                trend_reversal = True
            if position_side < 0 and bull_bias and chop_14[i] < 45:
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # If same side, keep position (don't reset highest/lowest)
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals