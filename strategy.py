#!/usr/bin/env python3
"""
Experiment #146: 12h Primary + 1d HTF — Simplified Regime-Adaptive Strategy

Hypothesis: Previous failures (#136, #139, #141, #142) had too many conflicting filters
resulting in 0 trades or negative Sharpe. This strategy uses SIMPLER logic:

1) 1d HMA(21) for macro trend bias — trade WITH the trend
2) Choppiness Index(14) for regime: >50 = range (CRSI mean revert), <50 = trend (Donchian)
3) TREND REGIME: Donchian(20) breakout in trend direction (simpler than volume filter)
4) RANGE REGIME: Connors RSI <15 long, >85 short (wider thresholds = more trades)
5) ATR(14) trailing stop at 2.5x — capital protection
6) Fewer AND conditions = more trades while maintaining quality

Why this should work on 12h:
- 12h naturally produces 20-50 trades/year (optimal fee/ratio)
- Simpler entry logic than failed #136, #139, #141, #142
- Wider CRSI thresholds (15/85 vs 20/80) = more mean reversion trades
- No volume filter (was killing trades in #141)
- 1d HTF provides stable trend bias without overfitting

Position size: 0.25 base, 0.30 with strong confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols (BTC, ETH, SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_simplified_regime_crsi_donchian_1d_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending.
    CHOP > 50 = range/choppy (mean revert), CHOP < 50 = trending
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.maximum(price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI - mean reversion indicator.
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    CRSI < 15 = oversold (long), CRSI > 85 = overbought (short)
    """
    close_s = pd.Series(close)
    
    # RSI(3)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(len(close))
    for i in range(1, len(close)):
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
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank(100) - where current return ranks vs last 100
    returns = close_s.pct_change()
    percent_rank = returns.rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100 if len(x) > 1 else 50,
        raw=False
    )
    percent_rank = percent_rank.fillna(50).values
    
    rsi_close_arr = rsi_close.fillna(50).values
    rsi_streak_arr = rsi_streak.fillna(50).values
    
    crsi = (rsi_close_arr + rsi_streak_arr + percent_rank) / 3.0
    return crsi

def calculate_rsi(close, period=14):
    """Calculate standard RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = np.inf
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        # CHOP > 50 = range/choppy (mean revert), CHOP < 50 = trending
        is_choppy = chop_14[i] > 50.0
        is_trending = chop_14[i] < 50.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TREND REGIME: Donchian Breakout ---
        if is_trending:
            # Long: price above 1d HMA + breaks Donchian high
            if price_above_hma_1d:
                if close[i] > donchian_upper[i-1]:
                    new_signal = POSITION_SIZE_BASE
                    # Strong confluence: RSI also bullish
                    if rsi_14[i] > 50.0:
                        new_signal = POSITION_SIZE_MAX
            
            # Short: price below 1d HMA + breaks Donchian low
            if price_below_hma_1d:
                if close[i] < donchian_lower[i-1]:
                    new_signal = -POSITION_SIZE_BASE
                    # Strong confluence: RSI also bearish
                    if rsi_14[i] < 50.0:
                        new_signal = -POSITION_SIZE_MAX
        
        # --- RANGE REGIME: CRSI Mean Reversion ---
        if is_choppy:
            # Long: CRSI < 15 (oversold) + price above or near 1d HMA
            if crsi[i] < 15.0:
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_BASE
                elif not price_below_hma_1d:
                    new_signal = POSITION_SIZE_BASE * 0.6  # weaker signal
            
            # Short: CRSI > 85 (overbought) + price below or near 1d HMA
            if crsi[i] > 85.0:
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_BASE
                elif not price_above_hma_1d:
                    new_signal = -POSITION_SIZE_BASE * 0.6  # weaker signal
        
        # === HOLD POSITION LOGIC ===
        # Hold existing position if no new signal and still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if above Donchian mid and CRSI not overbought
                if close[i] > donchian_mid[i] and crsi[i] < 80.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if below Donchian mid and CRSI not oversold
                if close[i] < donchian_mid[i] and crsi[i] > 20.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
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
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_1d and chop_14[i] < 50.0:  # trending reversal
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d and chop_14[i] < 50.0:  # trending reversal
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit in range regime) ===
        if in_position and position_side > 0 and crsi[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else np.inf
        else:
            if in_position and signals[i-1] != 0.0:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = np.inf
        
        signals[i] = new_signal
    
    return signals