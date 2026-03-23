#!/usr/bin/env python3
"""
Experiment #007: 1d Primary + 1w HTF — Connors RSI Mean Reversion with Regime Filter

Hypothesis: Connors RSI (CRSI) is a proven mean-reversion indicator with 75% win rate.
Combined with 1w HMA trend bias and Choppiness regime filter, this should work in
both bull and bear markets. The 2025 bear market requires mean-reversion entries,
not pure trend-following.

Key components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 + price > SMA200
   - Short: CRSI > 85 + price < SMA200
2. Choppiness Index: Only trade mean-reversion when CHOP > 50 (ranging)
3. 1w HMA: Macro trend bias for higher win rate
4. Donchian(20) breakout: Fallback for trending regimes
5. ATR(14) trailing stop: 2.5*ATR exit

Why this should work:
- CRSI catches oversold/overbought extremes better than standard RSI
- 1d timeframe = 20-50 trades/year (low fee drag)
- 1w HTF filter avoids counter-trend trades in strong moves
- LOOSE CRSI thresholds (15/85 not 10/90) ensure trade generation
- Dual entry (CRSI + Donchian) guarantees we get trades in all regimes

Position size: 0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_donchian_regime_1w_v1"
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

def calculate_rsi_streak(close, period=2):
    """
    Calculate RSI Streak component of Connors RSI.
    Measures consecutive up/down days as percentage.
    """
    n = len(close)
    streak_rsi = np.zeros(n)
    
    # Calculate streak values
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    for i in range(period, n):
        up_streaks = np.sum(streak[i-period+1:i+1] > 0)
        down_streaks = np.sum(streak[i-period+1:i+1] < 0)
        total = up_streaks + down_streaks
        if total > 0:
            streak_rsi[i] = (up_streaks / total) * 100.0
        else:
            streak_rsi[i] = 50.0
    
    return streak_rsi

def calculate_percent_rank(close, period=100):
    """
    Calculate Percent Rank component of Connors RSI.
    Percentage of past period days where close was lower than current.
    """
    n = len(close)
    pct_rank = np.zeros(n)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        count_lower = np.sum(window[:-1] < current)
        pct_rank[i] = (count_lower / (period - 1)) * 100.0
    
    return pct_rank

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    rsi_3 = calculate_rsi(close, period=rsi_period)
    streak_rsi = calculate_rsi_streak(close, period=streak_period)
    pct_rank = calculate_percent_rank(close, period=rank_period)
    
    crsi = (rsi_3 + streak_rsi + pct_rank) / 3.0
    return crsi

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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return highest, lowest

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    sma_200 = calculate_sma(close, period=200)
    donchian_high, donchian_low = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period + warmup
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(sma_200[i]):
            continue
        if np.isnan(donchian_high[i]) or atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Range market
        is_trending = chop_value < 45.0  # Trend market
        
        # === SMA200 FILTER ===
        price_above_sma200 = close[i] > sma_200[i]
        price_below_sma200 = close[i] < sma_200[i]
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 20.0  # Was 15, loosened to 20
        crsi_overbought = crsi[i] > 80.0  # Was 85, loosened to 80
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_high[i-1]  # Break above previous high
        donchian_breakout_short = close[i] < donchian_low[i-1]  # Break below previous low
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + above SMA200 + weekly bias helps
            if crsi_oversold and price_above_sma200:
                if price_above_hma_1w:  # Weekly trend confirmation (optional but helps)
                    new_signal = POSITION_SIZE
                else:
                    # Still enter if CRSI very extreme (< 10)
                    if crsi[i] < 10.0:
                        new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + below SMA200 + weekly bias helps
            elif crsi_overbought and price_below_sma200:
                if price_below_hma_1w:  # Weekly trend confirmation
                    new_signal = -POSITION_SIZE
                else:
                    # Still enter if CRSI very extreme (> 90)
                    if crsi[i] > 90.0:
                        new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Donchian Breakout ---
        elif is_trending:
            # Long: Donchian breakout + weekly confirms
            if donchian_breakout_long:
                if price_above_hma_1w or price_above_sma200:
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakout + weekly confirms
            elif donchian_breakout_short:
                if price_below_hma_1w or price_below_sma200:
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: Pure CRSI extremes (ensure we get trades) ---
        if new_signal == 0.0:
            # Very extreme CRSI regardless of regime
            if crsi[i] < 8.0 and price_above_sma200:
                new_signal = POSITION_SIZE
            elif crsi[i] > 92.0 and price_below_sma200:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
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
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if weekly trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1w and close[i] < sma_200[i]:
                new_signal = 0.0
        
        # Exit short if weekly trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1w and close[i] > sma_200[i]:
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