#!/usr/bin/env python3
"""
Experiment #019: 4h Primary + 1d HTF — Connors RSI + Choppiness Dual Regime

Hypothesis: Based on research showing Connors RSI (CRSI) works exceptionally well for 
mean reversion in crypto (75% win rate reported), combined with Choppiness Index regime 
filtering. This strategy uses:

1. CONNORS RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More sensitive than regular RSI, catches oversold/overbought extremes faster
   - Long when CRSI < 15, Short when CRSI > 85

2. CHOPPINESS INDEX regime filter:
   - CHOP > 61.8 = RANGING → Use CRSI mean reversion entries
   - CHOP < 38.2 = TRENDING → Use breakout entries with HMA confirmation
   - Transition zone = Stay flat or hold existing positions

3. 1d HMA trend bias:
   - Only long if price > 1d HMA (bullish bias)
   - Only short if price < 1d HMA (bearish bias)

4. Asymmetric position sizing:
   - With trend: 0.30 position size
   - Against trend (counter-trend mean reversion): 0.20 position size

5. ATR trailing stop: 2.5*ATR to protect capital

Why 4h works:
- Targets 30-60 trades/year (fee-efficient per Rule 10)
- Less noise than 1h/30m, more signals than 12h/1d
- Proven in experiment #016 that dual regime works at higher TF

Entry conditions (LOOSE enough to generate ≥10 trades/symbol):
- Long range: CRSI < 20 + CHOP > 55 + price > 1d HMA
- Short range: CRSI > 80 + CHOP > 55 + price < 1d HMA
- Long trend: Close > Donchian(20) high + CHOP < 45 + 4h HMA bullish
- Short trend: Close < Donchian(20) low + CHOP < 45 + 4h HMA bearish

Position size: 0.25-0.30 (discrete levels)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_crsi_chop_dual_regime_1d_v2"
timeframe = "4h"
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
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: Percentage of past N days where close was lower than today
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, period=streak_period)
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        count_lower = np.sum(close[i-rank_period:i] < close[i])
        percent_rank[i] = 100.0 * count_lower / rank_period
    
    # Combine components
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
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
    """Calculate Donchian Channel."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for regime bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # 4h HMA for trend confirmation
    hma_4h = calculate_hma(close, period=21)
    
    # Donchian channels for breakout detection
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_TREND = 0.30
    POSITION_SIZE_RANGE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(donchian_upper[i]):
            continue
        if atr_14[i] == 0 or np.isnan(hma_4h[i]):
            continue
        
        # === 1D TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H TREND CONFIRMATION ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-3] if i >= 3 else False
        price_above_hma_4h = close[i] > hma_4h[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Slightly lower threshold for more trades
        is_trending = chop_value < 45.0  # Slightly higher threshold for more trades
        is_transition = (chop_value >= 45.0) and (chop_value <= 55.0)
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i >= 1 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i >= 1 else False
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 20  # More lenient than 15 for more trades
        crsi_overbought = crsi[i] > 80  # More lenient than 85 for more trades
        
        # === DUAL REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with CRSI ---
        if is_ranging:
            # Long: CRSI oversold + price above 1d HMA (bullish bias)
            if crsi_oversold and price_above_hma_1d:
                new_signal = POSITION_SIZE_RANGE
            
            # Short: CRSI overbought + price below 1d HMA (bearish bias)
            elif crsi_overbought and price_below_hma_1d:
                new_signal = -POSITION_SIZE_RANGE
        
        # --- TRENDING REGIME: Breakout Following ---
        elif is_trending:
            # Long: Donchian breakout + 4h HMA bullish + price above 4h HMA
            if donchian_breakout_long and hma_4h_slope_bull and price_above_hma_4h:
                new_signal = POSITION_SIZE_TREND
            
            # Short: Donchian breakdown + 4h HMA bearish + price below 4h HMA
            elif donchian_breakout_short and hma_4h_slope_bear and price_below_hma_4h:
                new_signal = -POSITION_SIZE_TREND
        
        # --- TRANSITION REGIME: Hold existing, no new entries ---
        # new_signal remains 0.0 for new entries
        
        # === HOLD POSITION LOGIC ===
        # In transition zone, hold existing positions
        if in_position and is_transition and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # Also hold if no exit signal and in position
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
        # Exit long if regime changes from ranging to strong trending bearish
        if in_position and position_side > 0:
            if is_trending and hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        # Exit short if regime changes from ranging to strong trending bullish
        if in_position and position_side < 0:
            if is_trending and hma_4h_slope_bull and price_above_hma_4h:
                new_signal = 0.0
        
        # === EXIT ON CRSI REVERSAL (for mean reversion trades) ===
        if in_position and position_side > 0:
            if crsi[i] > 70:  # CRSI moved from oversold to overbought territory
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if crsi[i] < 30:  # CRSI moved from overbought to oversold territory
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
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
            if in_position and prev_signal != 0.0:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals