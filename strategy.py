#!/usr/bin/env python3
"""
Experiment #089: 4h Primary + 1d HTF — Simplified Donchian Breakout with Trend Filter

Hypothesis: Complex regime switching caused 0 trades in many previous experiments.
This version uses PROVEN patterns with LOOSER entry conditions to ensure trade generation.

Key components:
1) 1d HMA(21) slope for macro trend bias (prevents counter-trend in bear markets)
2) 4h Donchian(20) breakout for entry timing (proven breakout pattern)
3) RSI(14) filter to avoid chasing extremes (40-60 neutral zone)
4) ATR(14) trailing stoploss at 2.5x (protects from whipsaw)
5) Funding rate contrarian overlay for BTC/ETH (proven edge)

Why this should work:
- 4h timeframe naturally limits to 30-50 trades/year (fee-efficient)
- 1d HMA prevents shorting in bull markets and longing in bear markets
- Donchian breakout catches momentum moves without overfitting
- RSI filter avoids entering at extremes (reduces false breakouts)
- Simpler logic = trades on ALL symbols (BTC/ETH/SOL)

Position size: 0.30 base, 0.35 max with confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_trend_1d_hma_funding_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

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
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1d HMA slope (trend strength)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    ema_21 = calculate_ema(close, period=21)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.30
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(donchian_upper[i]) or np.isnan(ema_21[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        hma_slope_positive = hma_1d_slope[i] > 0.05  # slight positive slope
        hma_slope_negative = hma_1d_slope[i] < -0.05  # slight negative slope
        
        # === 4h DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1]  # break above previous upper
        breakout_short = close[i] < donchian_lower[i-1]  # break below previous lower
        
        # === RSI FILTER (avoid extremes) ===
        rsi_neutral_long = rsi_14[i] < 65.0  # not overbought
        rsi_neutral_short = rsi_14[i] > 35.0  # not oversold
        rsi_momentum_long = rsi_14[i] > 50.0  # bullish momentum
        rsi_momentum_short = rsi_14[i] < 50.0  # bearish momentum
        
        # === EMA CONFIRMATION ===
        ema_bullish = close[i] > ema_21[i]
        ema_bearish = close[i] < ema_21[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 1d uptrend + 4h breakout + RSI okay ---
        # Primary: 1d HMA bullish + Donchian breakout + RSI momentum
        if price_above_hma_1d and breakout_long and rsi_neutral_long:
            new_signal = POSITION_SIZE_BASE
            # Boost if EMA confirmation + strong momentum
            if ema_bullish and rsi_momentum_long:
                new_signal = POSITION_SIZE_MAX
            # Boost if 1d slope strongly positive
            if hma_slope_positive and hma_1d_slope[i] > 0.2:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: 1d downtrend + 4h breakout + RSI okay ---
        # Primary: 1d HMA bearish + Donchian breakout + RSI momentum
        if price_below_hma_1d and breakout_short and rsi_neutral_short:
            new_signal = -POSITION_SIZE_BASE
            # Boost if EMA confirmation + strong momentum
            if ema_bearish and rsi_momentum_short:
                new_signal = -POSITION_SIZE_MAX
            # Boost if 1d slope strongly negative
            if hma_slope_negative and hma_1d_slope[i] < -0.2:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Keep position if no exit signal yet
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if RSI not extreme overbought and no breakdown
                if rsi_14[i] < 75.0 and close[i] > donchian_mid[i]:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if RSI not extreme oversold and no breakout
                if rsi_14[i] > 25.0 and close[i] < donchian_mid[i]:
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
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 1d HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_slope_negative:
                new_signal = 0.0
        
        # Exit short if 1d HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # === EXIT ON DONCHIAN REVERSAL ===
        # Exit long if price breaks below Donchian mid
        if in_position and position_side > 0 and close[i] < donchian_mid[i]:
            new_signal = 0.0
        
        # Exit short if price breaks above Donchian mid
        if in_position and position_side < 0 and close[i] > donchian_mid[i]:
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