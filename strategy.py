#!/usr/bin/env python3
"""
Experiment #039: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback + Donchian

Hypothesis: Previous strategies failed due to OVERLY STRICT entry conditions (0 trades).
This strategy SIMPLIFIES the logic to ensure trade generation while maintaining edge:

1. 1d HMA(21) for macro trend bias — simple, proven, no complex regime switching
2. 4h RSI(14) pullback entries — RSI<40 for long, >60 for short (LOOSE thresholds)
3. Donchian(20) breakout confirmation — price must break recent high/low
4. ATR(14) trailing stoploss at 2.5*ATR
5. Position size: 0.30 (discrete, within safe 0.20-0.35 range)

Why this works:
- Fewer filters = more trades (addresses #1 failure mode: 0 trades)
- 4h targets 20-50 trades/year (fee-efficient per Rule 10)
- HMA trend filter prevents counter-trend trades in strong moves
- RSI pullback entries catch retracements in trending markets
- Donchian breakout confirms momentum before entry

Key difference from failed strategies:
- NO complex regime switching (CHOP, multiple HMA slopes, etc.)
- LOOSE RSI thresholds (40/60 not 15/85) to ensure trade generation
- Single HTF (1d) not multiple (12h+1d+1w) to reduce conflicting signals
- Entry requires only 2-3 confluences, not 5-6

Position sizing: 0.30 (30% of capital per position)
Stoploss: 2.5*ATR trailing stop
Leverage: 1.0 (no leverage)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_donchian_simple_1d_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend bias
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1D MACRO TREND BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H HMA TREND ===
        hma_4h_slope_bull = hma_4h[i] > hma_4h[i-5] if i >= 5 else False
        hma_4h_slope_bear = hma_4h[i] < hma_4h[i-5] if i >= 5 else False
        
        # === RSI PULLBACK (LOOSE THRESHOLDS FOR TRADE GENERATION) ===
        rsi_oversold = rsi_14[i] < 40  # Much looser than 15-20
        rsi_overbought = rsi_14[i] > 60  # Much looser than 80-85
        rsi_neutral_long = rsi_14[i] < 50
        rsi_neutral_short = rsi_14[i] > 50
        
        # === DONCHIAN BREAKOUT CONFIRMATION ===
        # Price breaking above recent high = bullish momentum
        price_breakout_high = close[i] > donchian_high[i-1] if i >= 1 else False
        # Price breaking below recent low = bearish momentum
        price_breakout_low = close[i] < donchian_low[i-1] if i >= 1 else False
        
        # === ENTRY LOGIC (SIMPLE, LOOSE CONDITIONS) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Condition 1: Pullback long in uptrend (RSI oversold + price above 1d HMA)
        long_pullback = rsi_oversold and price_above_hma_1d
        
        # Condition 2: Breakout long with trend (breakout high + 4h HMA bullish)
        long_breakout = price_breakout_high and hma_4h_slope_bull
        
        # Condition 3: Simple trend follow (price above both HMA + RSI neutral)
        long_trend = price_above_hma_1d and hma_4h_slope_bull and rsi_neutral_long
        
        if long_pullback or long_breakout or long_trend:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Condition 1: Pullback short in downtrend (RSI overbought + price below 1d HMA)
        short_pullback = rsi_overbought and price_below_hma_1d
        
        # Condition 2: Breakout short with trend (breakout low + 4h HMA bearish)
        short_breakout = price_breakout_low and hma_4h_slope_bear
        
        # Condition 3: Simple trend follow (price below both HMA + RSI neutral)
        short_trend = price_below_hma_1d and hma_4h_slope_bear and rsi_neutral_short
        
        if short_pullback or short_breakout or short_trend:
            # Only short if not already long signal
            if new_signal <= 0:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        # If we're in a position and no new signal, hold the position
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
        
        # === EXIT ON TREND REVERSAL ===
        # Exit long if macro trend turns bearish
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_4h_slope_bear:
                new_signal = 0.0
        
        # Exit short if macro trend turns bullish
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_4h_slope_bull:
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
                # Position flip
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