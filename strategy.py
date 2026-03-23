#!/usr/bin/env python3
"""
Experiment #014: 4h Primary + 12h/1d HTF — Ehlers Fisher Transform + Choppiness Regime

Hypothesis: Fisher Transform excels at catching reversals in bear/range markets (2022 crash, 2025 bear).
Combined with Choppiness Index regime filter and 12h/1d HMA trend bias, this should:
1. Catch reversal extremes better than RSI (Fisher normalizes to Gaussian distribution)
2. Avoid whipsaw in choppy markets via Choppiness filter
3. Align with macro trend via 12h/1d HMA bias
4. Generate 20-50 trades/year on 4h timeframe (appropriate for fee drag)

Key components:
1. Ehlers Fisher Transform (period=9): Long when Fisher crosses above -1.5, short when crosses below +1.5
2. Choppiness Index (14): >61.8 = range (use mean reversion), <38.2 = trend (use breakout)
3. 12h HMA(21) + 1d HMA(21): Macro trend bias (only trade with HTF trend)
4. ATR(14) trailing stop: 2.5*ATR for risk management
5. Donchian(20) breakout confirmation in trending regime

Why this should beat previous attempts:
- Fisher Transform is NEW (not tried on 4h yet) - research shows Sharpe 0.8-1.5 in bear markets
- Dual HTF (12h + 1d) provides stronger trend confirmation than single HTF
- Regime-adaptive: mean revert in chop, trend follow otherwise
- Loose Fisher thresholds (-1.5/+1.5 instead of -2/+2) ensure trade generation

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_donchian_regime_12h1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price into Gaussian distribution for better reversal detection.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            price_range = 1e-10
        
        # Normalize price to -1 to +1 range
        normalized = 0.66 * ((hl2 - lowest_low) / price_range - 0.5) + 0.67 * (
            0.66 * ((close[i] - lowest_low) / price_range - 0.5) + 0.67
        )
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher Transform
        fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Fisher Signal (previous Fisher value)
        fisher_signal[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, fisher_signal

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
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HMA for intermediate trend bias
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d HMA for macro trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(donchian_high[i]) or atr_14[i] == 0:
            continue
        
        # === 12H/1D MACRO BIAS ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # Strong bullish bias: both 12h and 1d HMA below price
        strong_bullish = price_above_hma_12h and price_above_hma_1d
        # Strong bearish bias: both 12h and 1d HMA above price
        strong_bearish = price_below_hma_12h and price_below_hma_1d
        # Neutral/mixed: HTF signals disagree
        neutral_bias = not strong_bullish and not strong_bearish
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 55.0  # Range market (mean reversion)
        is_trending = chop_value < 45.0  # Trend market (breakout)
        # Chop between 45-55 = transitional, use neutral logic
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_cross_up = fisher[i] > -1.5 and fisher_signal[i] <= -1.5  # Bullish reversal
        fisher_cross_down = fisher[i] < 1.5 and fisher_signal[i] >= 1.5  # Bearish reversal
        
        fisher_oversold = fisher[i] < -1.8  # Extreme oversold
        fisher_overbought = fisher[i] > 1.8  # Extreme overbought
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_up = close[i] > donchian_high[i-1]  # Break above previous high
        donchian_breakout_down = close[i] < donchian_low[i-1]  # Break below previous low
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_bullish = rsi_14[i] > 50.0
        rsi_bearish = rsi_14[i] < 50.0
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion with Fisher ---
        if is_ranging:
            # Long: Fisher oversold cross + RSI confirmation + HTF not strongly bearish
            if fisher_cross_up or fisher_oversold:
                if rsi_oversold or rsi_bullish:
                    if not strong_bearish:  # Avoid counter-trend in strong bear
                        new_signal = POSITION_SIZE
            
            # Short: Fisher overbought cross + RSI confirmation + HTF not strongly bullish
            elif fisher_cross_down or fisher_overbought:
                if rsi_overbought or rsi_bearish:
                    if not strong_bullish:  # Avoid counter-trend in strong bull
                        new_signal = -POSITION_SIZE
        
        # --- TRENDING REGIME: Breakout with HTF confirmation ---
        elif is_trending:
            # Long: Donchian breakout + strong bullish HTF bias + RSI confirmation
            if donchian_breakout_up:
                if strong_bullish and rsi_bullish:
                    new_signal = POSITION_SIZE
                elif neutral_bias and rsi_bullish and price_above_hma_12h:
                    new_signal = POSITION_SIZE * 0.5  # Half size in neutral
            
            # Short: Donchian breakout + strong bearish HTF bias + RSI confirmation
            elif donchian_breakout_down:
                if strong_bearish and rsi_bearish:
                    new_signal = -POSITION_SIZE
                elif neutral_bias and rsi_bearish and price_below_hma_12h:
                    new_signal = -POSITION_SIZE * 0.5  # Half size in neutral
        
        # --- TRANSITIONAL REGIME: Fisher + RSI confluence ---
        else:
            # Long: Fisher reversal + RSI oversold
            if fisher_cross_up and rsi_oversold:
                if not strong_bearish:
                    new_signal = POSITION_SIZE * 0.5
            
            # Short: Fisher reversal + RSI overbought
            elif fisher_cross_down and rsi_overbought:
                if not strong_bullish:
                    new_signal = -POSITION_SIZE * 0.5
        
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
        
        # === EXIT ON HTF TREND REVERSAL ===
        # Exit long if both 12h and 1d turn strongly bearish
        if in_position and position_side > 0:
            if strong_bearish and fisher[i] > 1.0:  # Fisher also turning
                new_signal = 0.0
        
        # Exit short if both 12h and 1d turn strongly bullish
        if in_position and position_side < 0:
            if strong_bullish and fisher[i] < -1.0:  # Fisher also turning
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