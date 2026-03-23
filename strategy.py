#!/usr/bin/env python3
"""
Experiment #092: 12h Primary + 1d/1w HTF — Dual Regime (Chop + CRSI) with Donchian Trend

Hypothesis: Previous 12h strategies failed because they used single-regime logic (always trend or always mean revert).
This version implements DUAL REGIME switching based on Choppiness Index:
- CHOP > 55 = Range regime → Connors RSI mean reversion (buy oversold, sell overbought)
- CHOP < 45 = Trend regime → Donchian breakout with HMA trend filter
- 45-55 = Transition → reduce position size or stay flat

Key innovations:
1) Connors RSI (CRSI) = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3 — proven 75% win rate
2) Dual regime based on Choppiness (not optional filter, but regime switch)
3) 1d HMA for intermediate trend bias, 1w HMA for macro trend (no shorts above 1w HMA)
4) Donchian(20) breakout only in trend regime with HMA confirmation
5) ATR(14) trailing stoploss at 3.0x (wider for 12h timeframe)
6) Position size: 0.30 base, 0.35 max with confluence, 0.15 in transition

Why this should beat #086:
- Regime-aware: doesn't force trend logic in ranging markets (2022-2023 chop)
- CRSI catches reversals better than simple RSI (research shows 0.8-1.5 Sharpe on BTC/ETH)
- 1w HMA prevents counter-trend shorts in bull markets
- Donchian breakout captures sustained moves in trend regime
- 12h naturally limits to 25-45 trades/year (fee-efficient)

Position size: 0.30 base, 0.35 max, 0.15 transition
Stoploss: 3.0*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_crsi_donchian_1d1w_v1"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentile rank of today's price change over last 100 days
    
    CRSI < 10 = extremely oversold (buy)
    CRSI > 90 = extremely overbought (sell)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_short = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI: consecutive up/down days
    delta = close_s.diff().fillna(0).values
    streak = np.zeros(n)
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if delta[i-1] > 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if delta[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, streak_abs[i] * 50 / streak_period)
        else:
            streak_rsi[i] = max(0, 100 - streak_abs[i] * 50 / streak_period)
    
    # Percent Rank: percentile of today's return over last rank_period days
    returns = close_s.pct_change().fillna(0).values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        if len(window) > 0:
            percent_rank[i] = 100.0 * np.sum(window <= returns[i]) / len(window)
        else:
            percent_rank[i] = 50.0
    
    # CRSI = average of three components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    crsi = np.nan_to_num(crsi, nan=50.0)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d HMA slope
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    hma_12h_21 = calculate_hma(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.30
    POSITION_SIZE_MAX = 0.35
    POSITION_SIZE_TRANSITION = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(hma_12h_21[i]):
            continue
        
        # === HTF TREND BIAS (1d and 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        hma_1d_slope_positive = hma_1d_slope[i] > 0.2
        hma_1d_slope_negative = hma_1d_slope[i] < -0.2
        
        # === CHOPPINESS REGIME ===
        chop_ranging = chop_14[i] > 55.0  # range regime
        chop_trending = chop_14[i] < 45.0  # trend regime
        chop_transition = not chop_ranging and not chop_trending  # 45-55
        
        # === CONNORS RSI SIGNALS (for range regime) ===
        crsi_oversold = crsi[i] < 15.0  # extreme oversold
        crsi_overbought = crsi[i] > 85.0  # extreme overbought
        crsi_neutral_long = crsi[i] < 50.0
        crsi_neutral_short = crsi[i] > 50.0
        
        # === DONCHIAN BREAKOUT (for trend regime) ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === HMA/EMA CONFIRMATION ===
        hma_bullish = hma_12h_21[i] > ema_50[i]
        hma_bearish = hma_12h_21[i] < ema_50[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGE REGIME: Connors RSI Mean Reversion ---
        if chop_ranging:
            # Long: CRSI oversold + price above 1w HMA (no counter-trend shorts in bull)
            if crsi_oversold and price_above_hma_1w:
                new_signal = POSITION_SIZE_BASE
                # Boost if also above 1d HMA
                if price_above_hma_1d:
                    new_signal = POSITION_SIZE_MAX
            
            # Short: CRSI overbought + price below 1w HMA (only short in bear macro)
            if crsi_overbought and price_below_hma_1w:
                new_signal = -POSITION_SIZE_BASE
                # Boost if also below 1d HMA
                if price_below_hma_1d:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- TREND REGIME: Donchian Breakout with HMA Filter ---
        elif chop_trending:
            # Long breakout: Donchian break + 1d HMA bullish + 1w HMA support
            if donchian_breakout_long and price_above_hma_1d and hma_bullish:
                new_signal = POSITION_SIZE_BASE
                # Boost if 1d slope positive
                if hma_1d_slope_positive:
                    new_signal = POSITION_SIZE_MAX
            
            # Short breakout: Donchian break + 1d HMA bearish + 1w HMA resistance
            if donchian_breakout_short and price_below_hma_1d and hma_bearish:
                new_signal = -POSITION_SIZE_BASE
                # Boost if 1d slope negative
                if hma_1d_slope_negative:
                    new_signal = -POSITION_SIZE_MAX
        
        # --- TRANSITION REGIME: Reduce size or flat ---
        elif chop_transition:
            # Only enter with strong confluence
            if crsi_oversold and price_above_hma_1w and price_above_hma_1d:
                new_signal = POSITION_SIZE_TRANSITION
            if crsi_overbought and price_below_hma_1w and price_below_hma_1d:
                new_signal = -POSITION_SIZE_TRANSITION
        
        # === HOLD POSITION LOGIC ===
        # Keep position if not at extreme exit
        if in_position and new_signal == 0.0:
            if position_side > 0 and crsi[i] < 80.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and crsi[i] > 20.0:
                new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (3.0 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 3.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 3.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 1d HMA turns bearish strongly
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_1d_slope_negative:
                new_signal = 0.0
        
        # Exit short if 1d HMA turns bullish strongly
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_1d_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON CRSI EXTREME (take profit in range regime) ===
        if chop_ranging:
            if in_position and position_side > 0 and crsi[i] > 75.0:
                new_signal = 0.0
            if in_position and position_side < 0 and crsi[i] < 25.0:
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