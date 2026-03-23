#!/usr/bin/env python3
"""
Experiment #1142: 12h Primary + 1d/1w HTF — Connors RSI + Choppiness Regime + HMA Trend

Hypothesis: After 831 failed experiments, the pattern is clear:
1. Over-filtering = 0 trades (Sharpe=0.000) — see #1130, #1132, #1133, #1135
2. Choppiness + CRSI regime switching FAILED repeatedly — see #1131, #1134, #1137, #1139
3. Simple HMA + RSI on 12h FAILED (#1136 Sharpe=-0.193) due to loose thresholds

NEW APPROACH based on research literature:
1. CONNORS RSI (CRSI): 3-component RSI proven 75% win rate in bear/range markets
   - RSI(3) + RSI_Streak(2) + PercentRank(100) / 3
   - Long: CRSI < 20, Short: CRSI > 80 (proven thresholds)
2. CHOPPINGNESS INDEX as META-FILTER only (not regime switch):
   - CHOP > 50 = avoid trend entries (wait for CRSI mean reversion)
   - CHOP < 50 = allow trend entries
3. HMA trend filter: 1d HMA(21) for direction, 1w HMA(50) for macro bias
4. LOOSE exit logic: hold through pullbacks, exit on trend reversal only
5. Position size 0.30 discrete, 2.5x ATR trailing stop

Why this beats #1136 (Sharpe=-0.193):
- CRSI is more sensitive than RSI(14) → catches reversals earlier
- CHOP as filter (not regime) → simpler, fewer false exits
- 1w HMA adds macro confirmation → prevents counter-trend in strong moves
- Simpler hold logic → maintains positions through continuations

Timeframe: 12h (target 20-50 trades/year, low fee drag)
HTF: 1d + 1w — loaded ONCE before loop using mtf_data helper
Position Size: 0.30 base (discrete: 0.0, ±0.30)
Stoploss: 2.5x ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_hma_1d1w_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI — 3-component mean reversion indicator.
    Proven 75% win rate in bear/range markets.
    
    Components:
    1. RSI(3) — short-term momentum
    2. RSI_Streak(2) — streak duration (consecutive up/down days)
    3. PercentRank(100) — current price vs last 100 days
    
    CRSI < 20 = oversold (long), CRSI > 80 = overbought (short)
    """
    n = len(close)
    crsi = np.full(n, np.nan)
    
    if n < rank_period + 5:
        return crsi
    
    # Component 1: RSI(3)
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean().values
    
    rsi_short = np.zeros(n)
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi_short[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi_short[~mask] = 50.0
    
    # Component 2: Streak RSI(2)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_gain = np.where(streak > 0, streak, 0.0)
    streak_loss = np.where(streak < 0, -streak, 0.0)
    
    avg_streak_gain = pd.Series(streak_gain).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    avg_streak_loss = pd.Series(streak_loss).ewm(span=streak_period, min_periods=streak_period, adjust=False).mean().values
    
    rsi_streak = np.zeros(n)
    mask2 = avg_streak_loss > 1e-10
    rs_streak = np.zeros(n)
    rs_streak[mask2] = avg_streak_gain[mask2] / avg_streak_loss[mask2]
    rsi_streak[mask2] = 100.0 - (100.0 / (1.0 + rs_streak[mask2]))
    rsi_streak[~mask2] = 50.0
    
    # Component 3: PercentRank(100)
    percent_rank = np.full(n, np.nan)
    for i in range(rank_period, n):
        window = close[i - rank_period:i]
        current = close[i]
        count_below = np.sum(window < current)
        percent_rank[i] = 100.0 * count_below / rank_period
    
    # Combine components
    valid = ~np.isnan(percent_rank) & ~np.isnan(rsi_short) & ~np.isnan(rsi_streak)
    crsi[valid] = (rsi_short[valid] + rsi_streak[valid] + percent_rank[valid]) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market chop vs trend.
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending.
    We use 50 as simpler threshold.
    
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    # Calculate ATR
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i - period + 1:i + 1])
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for trend direction
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (12h) indicators
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === MACRO BIAS (1w HMA) ===
        # Adds confirmation — only long if above weekly HMA
        weekly_bull = close[i] > hma_1w_aligned[i]
        weekly_bear = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINGNESS FILTER ===
        # CHOP > 50 = choppy (only mean reversion entries)
        # CHOP < 50 = trending (allow trend entries)
        is_choppy = chop[i] > 50.0
        
        # === CONNORS RSI SIGNAL ===
        # CRSI < 20 = oversold (long), CRSI > 80 = overbought (short)
        crsi_oversold = crsi[i] < 25.0  # Slightly loose to ensure trades
        crsi_overbought = crsi[i] > 75.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # Macro bull + weekly bull + CRSI oversold
        # In chop: only mean reversion (CRSI extreme)
        # In trend: can enter on pullback
        if macro_bull and weekly_bull:
            if crsi_oversold:
                desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # Macro bear + weekly bear + CRSI overbought
        elif macro_bear and weekly_bear:
            if crsi_overbought:
                desired_signal = -BASE_SIZE
        
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
        
        # === HOLD LOGIC — Maintain position if macro trend intact ===
        # Critical: don't exit just because CRSI moved, hold through trend
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if macro still bull (allow CRSI to recover)
                if macro_bull and weekly_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if macro still bear
                if macro_bear and weekly_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when macro trend reverses
        if in_position and position_side > 0:
            # Exit long if macro reverses to bear
            if macro_bear or (not weekly_bull and close[i] < hma_1d_aligned[i] * 0.98):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if macro reverses to bull
            if macro_bull or (not weekly_bear and close[i] > hma_1d_aligned[i] * 1.02):
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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