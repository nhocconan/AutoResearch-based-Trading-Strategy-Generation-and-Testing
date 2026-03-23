#!/usr/bin/env python3
"""
Experiment #026: 12h Primary + 1d HTF — Donchian Breakout + CRSI Timing + Choppiness Regime

Hypothesis: Combining the proven Donchian breakout from #016 (Sharpe=0.166 at 12h) with 
the CRSI entry timing from #024 (Sharpe=0.424 at 4h) should improve trade quality at 12h.

Key innovations:
1. DONCHIAN(20) BREAKOUT: Proven breakout signal that works across regimes
2. CONNORS RSI (CRSI): Entry timing — only enter breakouts when CRSI confirms momentum
3. CHOPPINESS INDEX: Regime filter — breakouts work better in trending (CHOP<45)
4. 1D HMA MACRO FILTER: Only trade breakouts in direction of daily trend
5. ASYMMETRIC SIZING: Larger positions when HTF confirms (0.30), smaller when against (0.20)

Why 12h works:
- Targets 20-50 trades/year (Rule 10 — fee efficient)
- Less noise than 4h, more signals than 1d
- Proven in #016 with Donchian + Choppiness

Entry conditions (LOOSE enough for trades):
- Long breakout: Price > Donchian(20) high + CRSI > 50 + CHOP < 55 + price > 1d HMA
- Short breakout: Price < Donchian(20) low + CRSI < 50 + CHOP < 55 + price < 1d HMA
- Long mean-revert: CRSI < 15 + CHOP > 55 + price > 1d HMA (ranging regime)
- Short mean-revert: CRSI > 85 + CHOP > 55 + price < 1d HMA (ranging regime)

Position size: 0.25 base, 0.30 with HTF confirmation (discrete levels)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_crsi_chop_regime_1d_v1"
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
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, period=rsi_period)
    
    # Streak calculation (consecutive up/down days)
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if returns.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif returns.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / (avg_streak_loss + 1e-10)
    rsi_streak = 100.0 - (100.0 / (1.0 + streak_rs))
    
    # PercentRank(100) - percentile of today's return over last 100 days
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period+1:i+1]
        if len(window_returns) > 0:
            current_return = returns.iloc[i]
            rank = np.sum(window_returns <= current_return) / len(window_returns)
            percent_rank[i] = rank * 100.0
    
    # CRSI = average of three components
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    return donchian_high, donchian_low, donchian_mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_CONFIRMED = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Middle threshold for regime switch
        is_trending = chop_value < 45.0
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20
        crsi_overbought = crsi[i] > 80
        crsi_bullish = crsi[i] > 50
        crsi_bearish = crsi[i] < 50
        crsi_neutral_low = crsi[i] < 40
        crsi_neutral_high = crsi[i] > 60
        
        # === DONCHIAN BREAKOUT ===
        price_above_donchian = close[i] > donchian_high[i-1]  # Break above previous high
        price_below_donchian = close[i] < donchian_low[i-1]   # Break below previous low
        
        # === POSITION SIZE BASED ON HTF CONFIRMATION ===
        pos_size = POSITION_SIZE_BASE
        if position_side > 0 and price_above_hma_1d:
            pos_size = POSITION_SIZE_CONFIRMED
        elif position_side < 0 and price_below_hma_1d:
            pos_size = POSITION_SIZE_CONFIRMED
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- TRENDING REGIME: Donchian Breakout ---
        if is_trending:
            # Long breakout: Price breaks Donchian high + CRSI bullish + 1d HMA confirms
            if price_above_donchian and crsi_bullish:
                if price_above_hma_1d:  # HTF confirmation
                    new_signal = POSITION_SIZE_CONFIRMED
                else:  # Against HTF trend - smaller size
                    new_signal = POSITION_SIZE_BASE
            
            # Short breakout: Price breaks Donchian low + CRSI bearish + 1d HMA confirms
            elif price_below_donchian and crsi_bearish:
                if price_below_hma_1d:  # HTF confirmation
                    new_signal = -POSITION_SIZE_CONFIRMED
                else:  # Against HTF trend - smaller size
                    new_signal = -POSITION_SIZE_BASE
        
        # --- RANGING REGIME: Mean Reversion ---
        elif is_ranging:
            # Long mean-revert: CRSI oversold + price above 1d HMA (bullish bias in range)
            if crsi_oversold and price_above_hma_1d:
                new_signal = POSITION_SIZE_BASE
            
            # Short mean-revert: CRSI overbought + price below 1d HMA (bearish bias in range)
            elif crsi_overbought and price_below_hma_1d:
                new_signal = -POSITION_SIZE_BASE
        
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
        # Exit long if regime changes from trending to ranging with bearish CRSI
        if in_position and position_side > 0:
            if is_ranging and crsi_overbought:
                new_signal = 0.0
        
        # Exit short if regime changes from trending to ranging with bullish CRSI
        if in_position and position_side < 0:
            if is_ranging and crsi_oversold:
                new_signal = 0.0
        
        # === EXIT ON HTF REVERSAL ===
        # Exit long if 1d HMA flips bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and crsi_bearish:
                new_signal = 0.0
        
        # Exit short if 1d HMA flips bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and crsi_bullish:
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