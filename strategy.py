#!/usr/bin/env python3
"""
Experiment #013: 1d Primary + 1w HTF — Connors RSI Mean Reversion + Choppiness Regime

Hypothesis: Connors RSI (CRSI) is proven to have 75% win rate in mean reversion scenarios.
Combined with Choppiness Index regime filter, this should excel in bear/range markets (2022, 2025)
while still capturing trends via Donchian breakout when CHOP < 38.2.

Key components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 (oversold extreme)
   - Short: CRSI > 85 (overbought extreme)
2. Choppiness Index: Regime detection
   - CHOP > 55 = range (use CRSI mean reversion)
   - CHOP < 45 = trend (use Donchian breakout)
3. 1w HMA: Macro trend bias (align trades with weekly direction)
4. Donchian(20): Breakout confirmation in trending regime
5. ATR(14) trailing stop: 2.5*ATR exit

Why this should beat previous attempts:
- CRSI is specifically designed for mean reversion (works in 2022 crash, 2025 bear)
- Loose CRSI thresholds (15/85 vs 10/90) ensure trade generation
- Dual regime logic adapts to market conditions
- 1d timeframe = 20-50 trades/year target (low fee drag)

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_chop_donchian_regime_1w_v2"
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

def calculate_crsi(close):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Percentage of past 100 days where close was lower than today
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, period=3)
    
    # RSI Streak (2) - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(2, n):
        if streak[i] > 0:
            # Up streak - calculate RSI of streak length
            streak_rsi[i] = 100.0 * (streak_abs[i] / (streak_abs[i] + 1))
        elif streak[i] < 0:
            # Down streak - inverse
            streak_rsi[i] = 100.0 * (1.0 / (streak_abs[i] + 1))
        else:
            streak_rsi[i] = 50.0
    
    # Smooth streak RSI with EMA(2)
    streak_rsi_s = pd.Series(streak_rsi).ewm(span=2, min_periods=2, adjust=False).mean().values
    
    # PercentRank(100) - percentage of past 100 closes lower than current
    percent_rank = np.zeros(n)
    lookback = 100
    for i in range(lookback, n):
        window = close[i-lookback:i]
        count_lower = np.sum(window < close[i])
        percent_rank[i] = 100.0 * count_lower / lookback
    
    # Fill early values
    percent_rank[:lookback] = 50.0
    
    # CRSI = average of three components
    crsi = (rsi_3 + streak_rsi_s + percent_rank) / 3.0
    
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

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
    crsi = calculate_crsi(close)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    # Calculate price position relative to Donchian
    donchian_position = np.zeros(n)
    for i in range(20, n):
        range_size = donchian_upper[i] - donchian_lower[i]
        if range_size > 0:
            donchian_position[i] = (close[i] - donchian_lower[i]) / range_size
        else:
            donchian_position[i] = 0.5
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(120, n):  # Need 100 for percentRank + 20 for Donchian
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market
        is_trending = chop_value < 45.0  # Trending market
        # 45-55 = transition zone (use either logic)
        
        # === CRSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 15.0  # Mean reversion long signal
        crsi_overbought = crsi[i] > 85.0  # Mean reversion short signal
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1]  # Break above previous high
        donchian_breakout_short = close[i] < donchian_lower[i-1]  # Break below previous low
        donchian_near_upper = close[i] > donchian_upper[i] * 0.98  # Within 2% of upper
        donchian_near_lower = close[i] < donchian_lower[i] * 1.02  # Within 2% of lower
        
        # === NEW SIGNAL LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: CRSI Mean Reversion ---
        if is_ranging:
            # Long: CRSI oversold + weekly bias helps OR neutral
            if crsi_oversold:
                if price_above_hma_1w or chop_value > 60:  # Weekly bullish OR very choppy
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + weekly bias helps OR neutral
            elif crsi_overbought:
                if price_below_hma_1w or chop_value > 60:  # Weekly bearish OR very choppy
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Donchian Breakout ---
        elif is_trending:
            # Long: Donchian breakout + weekly confirms
            if donchian_breakout_long:
                if price_above_hma_1w:  # Weekly trend confirmation
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakout + weekly confirms
            elif donchian_breakout_short:
                if price_below_hma_1w:  # Weekly trend confirmation
                    new_signal = -POSITION_SIZE
        
        # --- TRANSITION ZONE (45-55 CHOP): Hybrid approach ---
        else:
            # Use CRSI but with Donchian confirmation
            if crsi_oversold and donchian_near_lower:
                if price_above_hma_1w:
                    new_signal = POSITION_SIZE
            elif crsi_overbought and donchian_near_upper:
                if price_below_hma_1w:
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
        
        # === EXIT ON WEEKLY TREND REVERSAL ===
        # Exit long if weekly trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1w and chop_value < 40:  # Strong trend change
                new_signal = 0.0
        
        # Exit short if weekly trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1w and chop_value < 40:  # Strong trend change
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