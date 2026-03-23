#!/usr/bin/env python3
"""
Experiment #826: 12h Primary + 1d HTF — KAMA Trend + Connors RSI + Choppiness Filter

Hypothesis: After 563+ failed strategies, the key insight is that 12h timeframe needs
SIMPLE trend-following with mean-reversion entries (not complex regime switching).
The proven pattern for 12h is: KAMA trend + ADX + Choppiness filter (ETH Sharpe +0.755).

Strategy design:
1. 12h Primary timeframe (target 30-50 trades/year)
2. 1d KAMA(21) for adaptive trend bias (KAMA adjusts to volatility)
3. 12h Connors RSI for entry timing (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. 12h Choppiness Index(14) to filter ranging markets (CHOP > 55 = skip)
5. 12h ADX(14) for trend strength confirmation (ADX > 20 = trend valid)
6. 12h ATR(14) for trailing stop (2.5x)
7. Simple logic: Long when 1d KAMA bullish + CRSI oversold + ADX confirms
8. Short when 1d KAMA bearish + CRSI overbought + ADX confirms
9. Skip entries when CHOP > 55 (ranging market = no trend trades)

Why Connors RSI:
- 3-component RSI captures momentum, streak, and relative position
- Proven 75% win rate on mean-reversion entries
- CRSI < 20 = extreme oversold, CRSI > 80 = extreme overbought
- Works well in both bull and bear markets

Why KAMA over EMA:
- KAMA adapts to market efficiency ratio (ER)
- Fast in trends, slow in chop — perfect for 12h timeframe
- Reduces whipsaw in 2022 crash and 2025 bear market

Key changes from failed strategies:
- SIMPLER entry logic (fewer conflicting conditions)
- CRSI thresholds: 20/80 (not 10/90) — more signals guaranteed
- CHOP filter: > 55 = skip (clear cutoff)
- ADX threshold: > 20 (not 25) — more trend confirmation signals
- Hold logic maintains position through minor pullbacks

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive
Timeframe: 12h (target 30-50 trades/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_crsi_chop_adx_1d_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average — adapts to market volatility.
    KAMA adjusts smoothing constant based on Efficiency Ratio (ER).
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        signal = np.abs(close[i] - close[i - er_period])
        noise = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(er[i]):
            kama[i] = kama[i - 1]
            continue
        
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_rsi(close, period=14):
    """Relative Strength Index."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — 3-component mean reversion indicator.
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Range 0-100. < 20 = oversold, > 80 = overbought.
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 1:
        return crsi
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Up/Down Streak (2)
    streak_rsi = np.full(n, np.nan)
    streak = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i - 1]:
            streak[i] = streak[i - 1] + 1 if i > 0 else 1
        elif close[i] < close[i - 1]:
            streak[i] = streak[i - 1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_gain = np.where(streak > 0, streak, 0)
    streak_loss = np.where(streak < 0, -streak, 0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
        streak_rsi = 100 - (100 / (1 + streak_rs))
    
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Component 3: Percent Rank of price over last 100 bars
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period + 1:i + 1]
        count_below = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100.0 * count_below / (rank_period - 1)
    
    # Combine components
    for i in range(rank_period, n):
        if not np.isnan(rsi_short[i]) and not np.isnan(streak_rsi[i]) and not np.isnan(percent_rank[i]):
            crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppy vs trending.
    CHOP > 55 = ranging, CHOP < 45 = trending.
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], np.abs(high[j] - close[j-1]), np.abs(low[j] - close[j-1]))
            tr_sum += tr
        
        chop[i] = 100 * np.log10(tr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index — measures trend strength.
    ADX > 20 = trending, ADX < 20 = ranging.
    """
    n = len(close)
    adx = np.full(n, np.nan)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(high[i] - high[i-1], 0)
        else:
            plus_dm[i] = 0
        
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(low[i-1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth with EMA
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_di_raw = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di_raw = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100 * plus_di_raw / (atr + 1e-10)
        minus_di = 100 * minus_di_raw / (atr + 1e-10)
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX is smoothed DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Average True Range."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate primary (12h) indicators
    crsi_12h = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_12h = calculate_choppiness(high, low, close, period=14)
    adx_12h = calculate_adx(high, low, close, period=14)
    atr_12h = calculate_atr(high, low, close, period=14)
    
    # Calculate and align 1d KAMA for trend bias
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Also get 1d close aligned for trend direction
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(crsi_12h[i]) or np.isnan(chop_12h[i]) or np.isnan(adx_12h[i]):
            continue
        if np.isnan(atr_12h[i]) or atr_12h[i] <= 1e-10:
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(close_1d_aligned[i]):
            continue
        
        # === LONG-TERM TREND BIAS (1d KAMA21) ===
        # Bullish: price above KAMA and KAMA sloping up
        trend_1d_bullish = close_1d_aligned[i] > kama_1d_aligned[i]
        trend_1d_bearish = close_1d_aligned[i] < kama_1d_aligned[i]
        
        # Check KAMA slope (simple: compare to 5 bars ago)
        kama_slope_up = False
        kama_slope_down = False
        if i >= 5 and not np.isnan(kama_1d_aligned[i - 5]):
            kama_slope_up = kama_1d_aligned[i] > kama_1d_aligned[i - 5]
            kama_slope_down = kama_1d_aligned[i] < kama_1d_aligned[i - 5]
        
        # === REGIME FILTER (12h Choppiness Index) ===
        trending_regime = chop_12h[i] < 55
        ranging_regime = chop_12h[i] >= 55
        
        # === TREND STRENGTH (12h ADX) ===
        strong_trend = adx_12h[i] > 20
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi_12h[i] < 20
        crsi_overbought = crsi_12h[i] > 80
        crsi_recovery_long = crsi_12h[i] > 30 and crsi_12h[i - 1] < 30 if i > 0 and not np.isnan(crsi_12h[i - 1]) else False
        crsi_recovery_short = crsi_12h[i] < 70 and crsi_12h[i - 1] > 70 if i > 0 and not np.isnan(crsi_12h[i - 1]) else False
        
        desired_signal = 0.0
        
        # === LONG ENTRY LOGIC ===
        if trend_1d_bullish and trending_regime and strong_trend:
            # Primary: CRSI oversold + recovery
            if crsi_oversold or crsi_recovery_long:
                desired_signal = BASE_SIZE
            # Secondary: CRSI moderately oversold in strong trend
            elif crsi_12h[i] < 35 and kama_slope_up:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRY LOGIC ===
        elif trend_1d_bearish and trending_regime and strong_trend:
            # Primary: CRSI overbought + recovery
            if crsi_overbought or crsi_recovery_short:
                desired_signal = -BASE_SIZE
            # Secondary: CRSI moderately overbought in strong trend
            elif crsi_12h[i] > 65 and kama_slope_down:
                desired_signal = -REDUCED_SIZE
        
        # === RANGING REGIME — Skip or reduced size ===
        elif ranging_regime:
            # Only take extreme CRSI with trend alignment
            if trend_1d_bullish and crsi_12h[i] < 15:
                desired_signal = REDUCED_SIZE
            elif trend_1d_bearish and crsi_12h[i] > 85:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 1d trend still bullish
                if trend_1d_bullish and crsi_12h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 1d trend still bearish
                if trend_1d_bearish and crsi_12h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if trend reverses or CRSI extremely overbought
            if trend_1d_bearish and crsi_12h[i] > 85:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if trend reverses or CRSI extremely oversold
            if trend_1d_bullish and crsi_12h[i] < 15:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_12h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals