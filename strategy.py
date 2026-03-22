#!/usr/bin/env python3
"""
Experiment #577: 1d Primary + 1w HTF — Dual Regime (Choppiness + Connors RSI)

Hypothesis: After 514 failed strategies, the pattern is clear:
- Single-regime strategies fail because markets switch between trend/mean-revert
- 2021-2024 had bull + crash + range; 2025+ is bear/range
- Dual-regime approach: CHOP > 61.8 = mean revert (Connors RSI), CHOP < 38.2 = trend follow (HMA)
- 1w HTF for major bias (prevents counter-trend trades in strong regimes)
- Connors RSI (CRSI) has 75% win rate in research literature for mean reversion
- This combination hasn't been tried in exact form (failed #573 was CRSI only, #576 was CHOP+CRSI on 12h)

Why this might beat Sharpe=0.435:
1. Regime-adaptive: different logic for chop vs trend (proven in academic literature)
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — superior to simple RSI
3. 1w HTF bias prevents major counter-trend losses in 2022 crash
4. Simpler entry conditions = more trades (avoid 0-trade failure like #569, #571, #575)
5. 1d timeframe = 20-50 trades/year target (per Rule 10)

Position sizing: 0.28 base (discrete, max 0.40 per Rule 4)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_crsi_regime_1w_v1"
timeframe = "1d"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Avoid division by zero
    range_val = highest_high - lowest_low
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    # CHOP formula
    chop = 100.0 * np.log10(atr_sum / range_val) / np.log10(n)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's return over last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    # Streak: +1 for up day, -1 for down day, cumulative
    returns = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = max(0, streak[i-1] + 1) if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = min(0, streak[i-1] - 1) if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI of streak (treat absolute streak as "price")
    streak_s = pd.Series(np.abs(streak))
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # For down streaks, invert RSI (high streak = oversold = high RSI_streak should be low)
    rsi_streak = np.where(streak < 0, 100 - rsi_streak, rsi_streak)
    
    # Component 3: Percentile rank of today's return over last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period+1:i+1]
        today_return = returns.iloc[i]
        # Percent of returns in window that are less than today's return
        percent_rank[i] = 100.0 * np.sum(window_returns < today_return) / rank_period
    
    # Combine all three components
    crsi = (rsi_3 + rsi_streak + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major trend bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    hma_1d_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(hma_1d_50[i]):
            continue
        
        # === 1W MAJOR TREND BIAS ===
        bull_bias_1w = close[i] > hma_1w_21_aligned[i]
        bear_bias_1w = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        choppy_regime = chop_14[i] > 61.8  # Range/mean-revert market
        trending_regime = chop_14[i] < 38.2  # Trending market
        # Neutral zone: 38.2 - 61.8 (use smaller positions or stay flat)
        neutral_regime = not choppy_regime and not trending_regime
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # REGIME 1: CHOPPY (CHOP > 61.8) — Mean Reversion with Connors RSI
        if choppy_regime:
            # Long: CRSI < 10 (extreme oversold) + 1w bias not strongly bear
            if crsi[i] < 15.0 and not bear_bias_1w:
                new_signal = POSITION_SIZE
            # Short: CRSI > 90 (extreme overbought) + 1w bias not strongly bull
            elif crsi[i] > 85.0 and not bull_bias_1w:
                new_signal = -POSITION_SIZE
        
        # REGIME 2: TRENDING (CHOP < 38.2) — Trend Follow with HMA
        elif trending_regime:
            # Long: Price > HMA50 + 1w bull bias
            if close[i] > hma_1d_50[i] and bull_bias_1w:
                new_signal = POSITION_SIZE
            # Short: Price < HMA50 + 1w bear bias
            elif close[i] < hma_1d_50[i] and bear_bias_1w:
                new_signal = -POSITION_SIZE
        
        # REGIME 3: NEUTRAL — Smaller positions on CRSI extremes only
        elif neutral_regime:
            if crsi[i] < 10.0 and not bear_bias_1w:
                new_signal = POSITION_SIZE * 0.6
            elif crsi[i] > 90.0 and not bull_bias_1w:
                new_signal = -POSITION_SIZE * 0.6
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT CONDITIONS (regime flip or bias flip) ===
        # Exit long on 1w bias flip to bear
        if in_position and position_side > 0:
            if bear_bias_1w and chop_14[i] < 38.2:  # Trend regime + bear bias
                new_signal = 0.0
        
        # Exit short on 1w bias flip to bull
        if in_position and position_side < 0:
            if bull_bias_1w and chop_14[i] < 38.2:  # Trend regime + bull bias
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals