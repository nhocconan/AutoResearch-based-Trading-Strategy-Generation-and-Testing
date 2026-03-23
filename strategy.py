#!/usr/bin/env python3
"""
Experiment #026: 12h Primary + 1d HTF — Connors RSI + Choppiness Regime + Donchian Breakout

Hypothesis: 12h timeframe with daily trend bias should generate 20-50 trades/year with
improved Sharpe through regime-adaptive entries. Connors RSI (CRSI) captures short-term
mean reversion with 75%+ win rate, while Choppiness Index filters regime appropriately.

Key components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Extreme readings (<10 or >90) signal high-probability reversals
2. Choppiness Index (14): Regime detection (>55 = range, <40 = trend)
3. 1d HMA(21): Macro trend bias from higher timeframe
4. Donchian(20): Breakout confirmation for trend regime
5. ATR(14) trailing stop: 2.5*ATR risk management

Why this should work:
- 12h primary = moderate trade frequency (30-60/year target)
- 1d HTF = strong trend filter without excessive lag
- CRSI = proven mean reversion edge (Connors Research)
- Regime switching = adapt to bull/bear/range markets
- LOOSE CRSI thresholds (15/85 instead of 10/90) = ensure trade generation

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_crsi_chop_donchian_regime_1d_v1"
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
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Where current return ranks in last N periods
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # RSI Streak - measure consecutive up/down days
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
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = 50 + min(streak_abs[i] / streak_period, 1) * 50
        elif streak[i] < 0:
            streak_rsi[i] = 50 - min(streak_abs[i] / streak_period, 1) * 50
        else:
            streak_rsi[i] = 50
    streak_rsi = np.clip(streak_rsi, 0, 100)
    
    # Percent Rank - where does current return rank in last N periods
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) > 0:
            percent_rank[i] = np.sum(window <= returns[i]) / len(window) * 100
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    High CHOP (>61.8) = ranging, Low CHOP (<38.2) = trending
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (highest + lowest) / 2.0
    return highest, lowest, middle

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
    atr_7 = calculate_atr(high, low, close, period=7)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Also track price momentum
    momentum_5 = np.zeros(n)
    for i in range(5, n):
        momentum_5[i] = (close[i] - close[i-5]) / (close[i-5] + 1e-10) * 100
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
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
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(donchian_high[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_1d_slope = hma_1d_aligned[i] - hma_1d_aligned[i-5] if i >= 5 else 0
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # LOOSE threshold for more trades
        is_trending = chop_value < 45.0  # LOOSE threshold for more trades
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 20.0  # Was 10, now 20 for more longs
        crsi_overbought = crsi[i] > 80.0  # Was 90, now 80 for more shorts
        crsi_neutral_low = crsi[i] < 45.0
        crsi_neutral_high = crsi[i] > 55.0
        
        # === DONCHIAN BREAKOUT ===
        price_near_donchian_high = close[i] > donchian_high[i] * 0.995  # Near upper
        price_near_donchian_low = close[i] < donchian_low[i] * 1.005  # Near lower
        donchian_breakout_up = close[i] > donchian_high[i-1] if i > 0 else False
        donchian_breakout_down = close[i] < donchian_low[i-1] if i > 0 else False
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = close[i] < bb_lower[i] * 1.01
        price_near_bb_upper = close[i] > bb_upper[i] * 0.99
        bb_width = (bb_upper[i] - bb_lower[i]) / (bb_mid[i] + 1e-10)
        
        # === VOLATILITY FILTER ===
        vol_elevated = atr_7[i] > atr_14[i] * 1.05  # Recent vol above average
        vol_contracted = atr_7[i] < atr_14[i] * 0.95  # Vol contraction
        
        # === MOMENTUM FILTER ===
        mom_positive = momentum_5[i] > 0.5
        mom_negative = momentum_5[i] < -0.5
        
        # === ADAPTIVE REGIME ENTRY LOGIC (LOOSE CONDITIONS) ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion (CRSI-based) ---
        if is_ranging:
            # Long: CRSI oversold + near BB lower or Donchian low
            if crsi_oversold:
                if price_near_bb_lower or price_near_donchian_low or price_above_hma_1d:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + near BB upper or Donchian high
            elif crsi_overbought:
                if price_near_bb_upper or price_near_donchian_high or price_below_hma_1d:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following (Donchian breakout) ---
        elif is_trending:
            # Long: Donchian breakout up + CRSI not overbought + daily confirms
            if donchian_breakout_up and crsi_neutral_low:
                if price_above_hma_1d and hma_1d_slope > 0:
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakout down + CRSI not oversold + daily confirms
            elif donchian_breakout_down and crsi_neutral_high:
                if price_below_hma_1d and hma_1d_slope < 0:
                    new_signal = -POSITION_SIZE
            
            # Fallback: Pullback to Donchian mid in trend
            elif price_near_donchian_low and crsi_oversold:
                if price_above_hma_1d:  # Pullback in uptrend
                    new_signal = POSITION_SIZE
            elif price_near_donchian_high and crsi_overbought:
                if price_below_hma_1d:  # Pullback in downtrend
                    new_signal = -POSITION_SIZE
        
        # --- FALLBACK: Simple CRSI extreme if no regime signal ---
        if new_signal == 0.0:
            # Long: CRSI very oversold + momentum turning
            if crsi[i] < 25.0 and crsi[i] > crsi[i-1]:
                if mom_positive or price_above_hma_1d:
                    new_signal = POSITION_SIZE
            
            # Short: CRSI very overbought + momentum turning
            elif crsi[i] > 75.0 and crsi[i] < crsi[i-1]:
                if mom_negative or price_below_hma_1d:
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
        # Exit long if daily trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_1d_slope < -0.001 * hma_1d_aligned[i]:
                new_signal = 0.0
        
        # Exit short if daily trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_1d_slope > 0.001 * hma_1d_aligned[i]:
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