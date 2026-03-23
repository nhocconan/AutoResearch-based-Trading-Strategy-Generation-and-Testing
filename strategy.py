#!/usr/bin/env python3
"""
Experiment #426: 12h Primary + 1d HTF — Simplified Regime + KAMA Trend + CRSI Entries

Hypothesis: 12h timeframe with daily bias should produce 30-60 trades/year with
better risk-adjusted returns than #416 (Sharpe=-0.821). Key fixes from failures:
1. SIMPLER entry conditions — #416 had too many nested filters = 0 trades
2. KAMA (Kaufman Adaptive) instead of HMA — adapts to volatility, less whipsaw
3. Donchian breakout confirmation — ensures we catch real trend moves
4. More lenient CRSI thresholds (20/80 vs 15/85) — ensures adequate trade frequency
5. Single ATR trailing stop — let winners run, don't exit on every signal flip
6. 1d KAMA for bias (not HMA) — smoother trend filter

Why this should beat #416 and #417:
- 12h has less noise than 4h but more signals than 1d
- KAMA adapts to market conditions better than fixed-period HMA
- Donchian breakout ensures we enter on actual momentum, not just RSI extremes
- Fewer exit conditions = let profits run
- Simpler logic = more trades across all symbols (BTC, ETH, SOL)

Target: Sharpe > 0.612, 120-240 trades over 4-year train, DD < -35%
Timeframe: 12h | HTF: 1d | Position Size: 0.25-0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_crsi_donchian_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise — moves fast in trends, slow in chop.
    Efficiency Ratio (ER) = |close - close[n]| / sum(|close[i] - close[i-1]|)
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    # Efficiency Ratio
    er = np.zeros(n)
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = sum(abs(close[j] - close[j-1]) for j in range(i - er_period + 1, i + 1))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = high[i-period+1:i+1].max()
        lowest = low[i-period+1:i+1].min()
        
        if highest - lowest < 1e-10:
            chop[i] = 50.0
            continue
        
        tr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            tr_sum += tr
        
        chop[i] = 100.0 * np.log10(tr_sum / (highest - lowest)) / np.log10(period)
        chop[i] = np.clip(chop[i], 0, 100)
    
    return chop

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    # RSI(3)
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
    rsi_3 = 100.0 - (100.0 / (1.0 + rs))
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Percent Rank
    for i in range(rank_period, n):
        window = close[i-rank_period:i]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        crsi[i] = (rsi_3.iloc[i] + rsi_streak.iloc[i] + rank) / 3.0
    
    return crsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(close)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period, n):
        upper[i] = high[i-period+1:i+1].max()
        lower[i] = low[i-period+1:i+1].min()
    
    return upper, lower

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    # Recalculate kama_50 with different parameters
    kama_50 = np.full(n, np.nan)
    er_period = 10
    for i in range(er_period, n):
        signal = abs(close[i] - close[i - er_period])
        noise = sum(abs(close[j] - close[j-1]) for j in range(i - er_period + 1, i + 1))
        er = signal / noise if noise > 1e-10 else 0.0
        fast_sc = 2.0 / 3
        slow_sc = 2.0 / 31
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        if i == er_period:
            kama_50[i] = close[i]
        else:
            kama_50[i] = kama_50[i-1] + sc * (close[i] - kama_50[i-1])
    
    chop = calculate_choppiness(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    # Calculate and align HTF KAMA for bias (1d)
    kama_1d_raw = calculate_kama(df_1d['close'].values, er_period=10, fast_period=2, slow_period=30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[100:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% position size for 12h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(chop[i]) or np.isnan(crsi[i]):
            continue
        if np.isnan(kama_1d_aligned[i]) or np.isnan(kama_21[i]) or np.isnan(kama_50[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 55.0  # Slightly lower threshold for more trades
        is_trending = chop[i] < 45.0
        
        # === HTF BIAS (1d KAMA) ===
        price_above_kama_1d = close[i] > kama_1d_aligned[i]
        price_below_kama_1d = close[i] < kama_1d_aligned[i]
        
        # === PRIMARY TREND (12h KAMA crossover) ===
        kama_bullish = kama_21[i] > kama_50[i]
        kama_bearish = kama_21[i] < kama_50[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] >= donchian_upper[i] * 0.998  # Near upper band
        donchian_breakout_short = close[i] <= donchian_lower[i] * 1.002  # Near lower band
        
        # === CONNORS RSI THRESHOLDS (more lenient for trade frequency) ===
        crsi_oversold = crsi[i] < 20.0  # Long signal
        crsi_overbought = crsi[i] > 80.0  # Short signal
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.8:
            position_size = BASE_SIZE * 0.7
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # LONG SETUP — Simplified conditions for more trades
        if price_above_kama_1d:  # HTF bullish bias
            if is_trending and kama_bullish:
                # Trend following — enter on pullback OR breakout
                if crsi_oversold or donchian_breakout_long:
                    desired_signal = position_size
            elif is_choppy:
                # Mean reversion in range
                if crsi_oversold:
                    desired_signal = position_size
            elif kama_bullish:
                # KAMA bullish with CRSI pullback
                if crsi_oversold or donchian_breakout_long:
                    desired_signal = position_size
        
        # SHORT SETUP — Simplified conditions for more trades
        if price_below_kama_1d:  # HTF bearish bias
            if is_trending and kama_bearish:
                # Trend following — enter on rally OR breakdown
                if crsi_overbought or donchian_breakout_short:
                    desired_signal = -position_size
            elif is_choppy:
                # Mean reversion in range
                if crsi_overbought:
                    desired_signal = -position_size
            elif kama_bearish:
                # KAMA bearish with CRSI rally
                if crsi_overbought or donchian_breakout_short:
                    desired_signal = -position_size
        
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
        
        # === CRSI EXTREME EXIT (take profit) ===
        if in_position and position_side > 0 and crsi[i] > 85.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and crsi[i] < 15.0:
            desired_signal = 0.0
        
        # === HTF BIAS REVERSAL EXIT ===
        if in_position and position_side > 0 and price_below_kama_1d:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and price_above_kama_1d:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and price_above_kama_1d:
                desired_signal = position_size
            elif position_side < 0 and price_below_kama_1d:
                desired_signal = -position_size
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
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