#!/usr/bin/env python3
"""
Experiment #030: 1h Primary + 4h/12h HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: 1h timeframe with Connors RSI (more responsive than standard RSI) + 
Choppiness Index regime + 4h/12h HMA trend bias will generate 40-80 trades/year
with positive Sharpe. Key insight: lower TF needs LOOSE entry conditions to 
guarantee trades while HTF provides directional bias.

Strategy Logic:
1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — faster than RSI(14)
2. CHOPPINESS INDEX: CHOP > 55 = range (mean revert), CHOP < 45 = trend (follow)
3. 4h HMA(21): Primary trend bias (trade with 4h direction)
4. 12h HMA(21): Macro confirmation (avoid counter-macro trades)
5. Entry: CRSI < 15 (long) or CRSI > 85 (short) + HTF alignment
6. Exit: CRSI cross 50 or stoploss at 2.5*ATR

Why this should work:
- CRSI is more responsive than RSI(14) — catches reversals faster
- 1h primary = more entry opportunities than 4h/12h
- 4h/12h HTF = avoids counter-trend trades that kill Sharpe
- LOOSE CRSI thresholds (15/85 not 10/90) = ensures trade generation
- Discrete sizing (0.25) = minimizes fee churn

Position size: 0.25 (smaller for 1h due to more trades)
Stoploss: 2.5*ATR trailing
Target trades: 50-80/year on 1h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_crsi_chop_regime_4h12h_v1"
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
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Component 2: RSI of streak length
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
    
    # RSI of streak (treat streak as price series)
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # Component 3: Percent Rank over lookback period
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        lookback = close[i-rank_period:i]
        current = close[i]
        count_below = np.sum(lookback < current)
        percent_rank[i] = count_below / rank_period * 100.0
    
    # CRSI = average of 3 components
    crsi = (rsi_3 + rsi_streak.values + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for primary trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 12h HMA for macro confirmation
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40, smaller for 1h)
    POSITION_SIZE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H PRIMARY TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 12H MACRO CONFIRMATION ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 52.0  # Range market (mean revert)
        is_trending = chop_value < 48.0  # Trend market (follow)
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 18.0  # Long entry
        crsi_overbought = crsi[i] > 82.0  # Short entry
        crsi_neutral = 45.0 < crsi[i] < 55.0  # Exit zone
        
        # === CRSI TURNING POINTS ===
        crsi_rising = crsi[i] > crsi[i-1] if i > 0 else False
        crsi_falling = crsi[i] < crsi[i-1] if i > 0 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion (primary mode) ---
        if is_ranging:
            # Long: CRSI oversold + 4h trend helps (not required, just helps)
            if crsi_oversold:
                # Enter if 4h is neutral or bullish (avoid strong bearish macro)
                if price_above_hma_4h or not price_below_hma_12h:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + 4h trend helps
            elif crsi_overbought:
                # Enter if 4h is neutral or bearish (avoid strong bullish macro)
                if price_below_hma_4h or not price_above_hma_12h:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Follow 4h direction with CRSI pullback ---
        elif is_trending:
            # Long: 4h bullish + CRSI pullback (not extreme)
            if price_above_hma_4h and crsi[i] < 40.0 and crsi_rising:
                if price_above_hma_12h:  # 12h confirms
                    new_signal = POSITION_SIZE
            
            # Short: 4h bearish + CRSI bounce (not extreme)
            elif price_below_hma_4h and crsi[i] > 60.0 and crsi_falling:
                if price_below_hma_12h:  # 12h confirms
                    new_signal = -POSITION_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit long when CRSI reaches neutral/overbought
        if in_position and position_side > 0:
            if crsi[i] > 55.0 or crsi_overbought:
                new_signal = 0.0
        
        # Exit short when CRSI reaches neutral/oversold
        if in_position and position_side < 0:
            if crsi[i] < 45.0 or crsi_oversold:
                new_signal = 0.0
        
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
        
        # === HOLD POSITION LOGIC ===
        # If already in position and no exit signal, hold
        if in_position and new_signal == 0.0 and not stoploss_triggered:
            # Check if we should maintain position
            if position_side > 0 and not crsi_overbought and crsi[i] > 30:
                new_signal = POSITION_SIZE
            elif position_side < 0 and not crsi_oversold and crsi[i] < 70:
                new_signal = -POSITION_SIZE
        
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