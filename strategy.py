#!/usr/bin/env python3
"""
Experiment #046: 12h Primary + 1d HTF — Dual Regime with Adaptive Entry

Hypothesis: Based on experiment history showing 12h/1d strategies have positive Sharpe 
(exp #037: 0.176, #042: 0.240, #043: 0.219), I'm combining regime-adaptive logic with 
loose entry conditions to ensure sufficient trade generation.

Key innovations:
1. CHOPPINESS INDEX regime detection: CHOP > 55 = range (mean revert), CHOP < 40 = trend
2. CONNORS RSI for mean reversion entries in ranging markets (75% win rate per research)
3. DONCHIAN breakout for trending markets with HMA confirmation
4. VOLATILITY COMPRESSION filter: BB Width < 30th percentile = coiling before move
5. LOOSE entry thresholds to guarantee 10+ trades per symbol (learning from 0-trade failures)

Why 12h works:
- Targets 20-50 trades/year (Rule 10 fee-efficient range)
- Less noise than 1h/4h, more signals than 1d
- Proven in exp #042 (Sharpe=0.240) and #043 (Sharpe=0.219)

Entry conditions (LOOSE to generate trades):
- Long range: CRSI < 30 + CHOP > 50 + price > 1d HMA
- Short range: CRSI > 70 + CHOP > 50 + price < 1d HMA
- Long trend: Price > Donchian(20) high + 12h HMA bullish + 1d HMA bullish
- Short trend: Price < Donchian(20) low + 12h HMA bearish + 1d HMA bearish

Position size: 0.28 (discrete, within 0.20-0.35 range per Rule 4)
Stoploss: 2.5*ATR trailing stop (signal → 0)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_crsi_donchian_regime_1d_v1"
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
    
    # Streak calculation
    returns = close_s.pct_change()
    streak = np.zeros(n)
    for i in range(1, n):
        if pd.isna(returns.iloc[i]):
            streak[i] = 0
        elif returns.iloc[i] > 0:
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
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window_returns = returns.iloc[i-rank_period+1:i+1].dropna()
        if len(window_returns) > 0:
            current_return = returns.iloc[i]
            if pd.isna(current_return):
                percent_rank[i] = 50.0
            else:
                rank = np.sum(window_returns <= current_return) / len(window_returns)
                percent_rank[i] = rank * 100.0
    
    crsi = (rsi_close + rsi_streak.values + percent_rank) / 3.0
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
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
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_bb_width(close, high, low, period=20, std_mult=2.0):
    """Calculate Bollinger Band Width (normalized)."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    width = (std_mult * 2 * std) / (sma + 1e-10)
    return width

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
    
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    hma_12h = calculate_hma(close, period=21)
    
    bb_width = calculate_bb_width(close, high, low, period=20, std_mult=2.0)
    # Calculate BB width percentile for squeeze detection
    bb_width_percentile = np.zeros(n)
    for i in range(100, n):
        window = bb_width[i-100:i+1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            bb_width_percentile[i] = np.sum(valid_window <= bb_width[i]) / len(valid_window) * 100
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(bb_width_percentile[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === VOLATILITY CONTEXT ===
        vol_expansion = (atr_7[i] / atr_14[i]) > 1.3  # Recent vol > average
        bb_squeeze = bb_width_percentile[i] < 35  # BB width in bottom 35%
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 12H TREND BIAS ===
        hma_12h_slope_bull = hma_12h[i] > hma_12h[i-5] if i >= 5 else False
        hma_12h_slope_bear = hma_12h[i] < hma_12h[i-5] if i >= 5 else False
        price_above_hma_12h = close[i] > hma_12h[i]
        price_below_hma_12h = close[i] < hma_12h[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0  # Lower threshold for more range trades
        is_trending = chop_value < 42.0  # Higher threshold for more trend trades
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 30  # Loose threshold
        crsi_overbought = crsi[i] > 70  # Loose threshold
        crsi_neutral_low = crsi[i] < 45
        crsi_neutral_high = crsi[i] > 55
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion ---
        if is_ranging:
            # Long: CRSI oversold + macro support (LOOSE conditions)
            if crsi_oversold:
                if price_above_hma_1d or bb_squeeze:  # Either macro bias OR squeeze
                    new_signal = POSITION_SIZE
            
            # Short: CRSI overbought + macro resistance
            elif crsi_overbought:
                if price_below_hma_1d or bb_squeeze:
                    new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Trend Following ---
        elif is_trending:
            # Long: Donchian breakout + trend confirmation
            if donchian_breakout_long:
                if hma_12h_slope_bull and (price_above_hma_1d or vol_expansion):
                    new_signal = POSITION_SIZE
            
            # Short: Donchian breakdown + trend confirmation
            elif donchian_breakout_short:
                if hma_12h_slope_bear and (price_below_hma_1d or vol_expansion):
                    new_signal = -POSITION_SIZE
        
        # --- SQUEEZE BREAKOUT (works in any regime) ---
        if bb_squeeze and vol_expansion:
            if donchian_breakout_long and price_above_hma_1d:
                new_signal = POSITION_SIZE
            elif donchian_breakout_short and price_below_hma_1d:
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
        if in_position and position_side > 0:
            # Exit long if strong bearish trend emerges
            if is_trending and hma_12h_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if strong bullish trend emerges
            if is_trending and hma_12h_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals