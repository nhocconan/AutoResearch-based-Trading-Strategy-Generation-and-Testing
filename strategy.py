#!/usr/bin/env python3
"""
Experiment #622: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + Asymmetric Entries

Hypothesis: Building on mtf_1d_chop_crsi_regime_1w_v1 (Sharpe=0.520), this strategy moves to 
12h primary timeframe with 1d HTF filter. Lower frequency = fewer trades = less fee drag.
Key innovation: Connors RSI (more responsive than standard RSI) + Choppiness regime switching
with ASYMMETRIC thresholds that match crypto behavior (fast crashes, slow rallies).

Why this might beat Sharpe=0.520:
1. 12h TF reduces noise vs 1d while maintaining sufficient trade frequency (30-50/year)
2. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — catches reversals faster
3. Choppiness Index cleanly separates trend vs range regimes
4. Asymmetric entries: longs need CRSI<25 (deep), shorts need CRSI>70 (moderate)
5. 1d HTF KAMA slope prevents counter-trend trades during major moves
6. Conservative size (0.30) with 2.5*ATR trailing stop controls 2022-style crashes

Position sizing: 0.30 discrete (per Rule 4, max 0.40)
Target: 30-50 trades/year on 12h (per Rule 10)
Stoploss: 2.5*ATR trailing via signal→0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_asym_1d_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Components:
    1. RSI(3) on close — short-term momentum
    2. RSI(2) on up/down streak — streak strength
    3. PercentRank(100) — where current return ranks vs last 100 bars
    
    CRSI < 10 = extremely oversold (long signal)
    CRSI > 90 = extremely overbought (short signal)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI(2) on streak
    # Streak: consecutive up/down days
    delta = close_s.diff()
    streak = np.zeros(n)
    for i in range(1, n):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI on streak (treat positive streak as gains, negative as losses)
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    streak_gain_s = pd.Series(streak_gain)
    streak_loss_s = pd.Series(streak_loss)
    
    avg_streak_gain = streak_gain_s.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss_s.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs_streak = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + rs_streak))
    rsi_streak = rsi_streak.values
    
    # Component 3: PercentRank(100) — current return rank vs last 100
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        current = returns[i]
        rank = np.sum(window < current) / pr_period
        percent_rank[i] = rank * 100.0
    
    percent_rank[:pr_period] = 50.0
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3.0
    crsi = np.clip(crsi, 0.0, 100.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean reversion works)
    CHOP < 38.2 = trending market (trend following works)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, 14)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    price_range = highest_high - lowest_low
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(n)
    
    chop = np.clip(chop, 0.0, 100.0)
    chop = np.nan_to_num(chop, nan=50.0)
    
    return chop

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Price change over ER period
    price_change = np.abs(close_s.diff(er_period).values)
    
    # Sum of absolute price changes (volatility)
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    # Efficiency Ratio (ER)
    with np.errstate(divide='ignore', invalid='ignore'):
        er = price_change / (volatility + 1e-10)
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0.0, 1.0)
    
    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Adaptive smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    kama[:er_period] = np.nan
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d KAMA for primary trend direction
    kama_1d = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate 12h indicators
    kama_12h = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(kama_12h[i]) or np.isnan(kama_1d_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(atr_14[i]) or np.isnan(crsi[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D TREND BIAS (KAMA slope over 3 bars) ===
        kama_1d_slope_bull = kama_1d_aligned[i] > kama_1d_aligned[i-3] if i >= 3 else False
        kama_1d_slope_bear = kama_1d_aligned[i] < kama_1d_aligned[i-3] if i >= 3 else False
        
        # Price relative to 1d KAMA
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === 12H KAMA SLOPE (2 bars) ===
        kama_12h_slope_bull = kama_12h[i] > kama_12h[i-2] if i >= 2 else False
        kama_12h_slope_bear = kama_12h[i] < kama_12h[i-2] if i >= 2 else False
        
        # Price relative to 12h KAMA
        price_above_kama_12h = close[i] > kama_12h[i]
        price_below_kama_12h = close[i] < kama_12h[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_trend_regime = chop_14[i] < 45.0
        is_chop_regime = chop_14[i] > 55.0
        
        # === ASYMMETRIC ENTRY LOGIC (looser thresholds for more trades) ===
        new_signal = 0.0
        
        # --- TREND REGIME: Follow 1d trend with 12h pullback entries ---
        if is_trend_regime:
            # LONG: 1d bull + 12h bull + price above both KAMAs + CRSI pullback (25-55)
            if kama_1d_slope_bull and kama_12h_slope_bull and price_above_kama_1d and price_above_kama_12h:
                if crsi[i] < 55.0:  # Looser: any pullback in uptrend
                    new_signal = POSITION_SIZE
            
            # SHORT: 1d bear + 12h bear + price below both KAMAs + CRSI bounce (45-75)
            elif kama_1d_slope_bear and kama_12h_slope_bear and price_below_kama_1d and price_below_kama_12h:
                if crsi[i] > 45.0:  # Looser: any bounce in downtrend
                    new_signal = -POSITION_SIZE
        
        # --- CHOP REGIME: Mean reversion at CRSI extremes (asymmetric) ---
        elif is_chop_regime:
            # LONG: CRSI < 30 (oversold) — asymmetric, crypto drops fast
            if crsi[i] < 30.0:
                new_signal = POSITION_SIZE
            
            # SHORT: CRSI > 70 (overbought) — asymmetric, rallies slower
            elif crsi[i] > 70.0:
                new_signal = -POSITION_SIZE
        
        # --- NEUTRAL REGIME (45-55): Wait for extremes only ---
        else:
            # Only enter on very extreme CRSI
            if crsi[i] < 15.0:
                new_signal = POSITION_SIZE
            elif crsi[i] > 85.0:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC (don't flip unless signal changes) ===
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if kama_1d_slope_bear and price_below_kama_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if kama_1d_slope_bull and price_above_kama_1d:
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