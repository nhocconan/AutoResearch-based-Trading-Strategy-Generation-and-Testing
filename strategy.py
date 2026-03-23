#!/usr/bin/env python3
"""
Experiment #627: 1d Primary + 12h HTF — Dual HMA Crossover + RSI Filter + ATR Stop

Hypothesis: Daily timeframe with 12h trend filter provides cleaner signals than 4h.
Building on the 4h version (#624) which had borderline Sharpe (-0.002), this uses:
- 1d as primary (fewer false signals, lower fee drag)
- 12h HMA slope for trend bias (faster response than 1w)
- HMA(21)/HMA(9) crossover for entry timing
- RSI(14) filter to avoid extreme entries
- 2.5*ATR trailing stop for risk management

Why this might work better than #624:
- 1d has less noise than 4h (fewer whipsaws in 2022 crash)
- 12h HTF is faster than 1w (more responsive to trend changes)
- Simpler entry logic (no Donchian, just HMA cross + RSI)
- Relaxed RSI filter (30-70 vs 40-60) = more trades
- Conservative size 0.30 controls drawdown

Key insight from 554 failed strategies:
- Over-engineered filters = 0 trades (#615, #619, #620, #621)
- 1w HTF too slow for many setups (#621, #623)
- Simple HMA crossover with RSI filter worked on SOL (Sharpe +0.879)
- 1d timeframe reduces fee drag significantly vs 4h

Target: 25-40 trades/year on 1d (per Rule 10)
Position sizing: 0.30 discrete
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_cross_rsi_12h_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    Faster response than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = max(1, int(period / 2))
    sqrt_n = max(1, int(np.sqrt(period)))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 12h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for primary trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1d indicators
    hma_21 = calculate_hma(close, period=21)
    hma_9 = calculate_hma(close, period=9)
    hma_48 = calculate_hma(close, period=48)
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
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
        if np.isnan(hma_21[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(rsi_14[i]):
            continue
        if np.isnan(hma_9[i]) or np.isnan(hma_48[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (HMA slope over 3 bars) ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3]
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3]
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === 1D HMA FAST/SLOW CROSSOVER ===
        hma_cross_bull = hma_9[i] > hma_21[i]
        hma_cross_bear = hma_9[i] < hma_21[i]
        
        # === 1D HMA SLOPE (2 bars) ===
        hma_21_slope_bull = hma_21[i] > hma_21[i-2]
        hma_21_slope_bear = hma_21[i] < hma_21[i-2]
        
        # === RSI FILTER (avoid extremes) ===
        rsi_not_overbought = rsi_14[i] < 70.0
        rsi_not_oversold = rsi_14[i] > 30.0
        rsi_bullish_zone = rsi_14[i] > 45.0
        rsi_bearish_zone = rsi_14[i] < 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 12h bull trend + 1d HMA cross + RSI confirmation ---
        # Condition 1: 12h HMA sloping up + price above 12h HMA (trend bias)
        # Condition 2: 1d HMA(9) > HMA(21) (momentum cross)
        # Condition 3: RSI not overbought (<70) and in bullish zone (>45)
        if hma_12h_slope_bull and price_above_hma_12h:
            if hma_cross_bull and rsi_not_overbought and rsi_bullish_zone:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: 12h bear trend + 1d HMA cross + RSI confirmation ---
        # Condition 1: 12h HMA sloping down + price below 12h HMA (trend bias)
        # Condition 2: 1d HMA(9) < HMA(21) (momentum cross)
        # Condition 3: RSI not oversold (>30) and in bearish zone (<55)
        elif hma_12h_slope_bear and price_below_hma_12h:
            if hma_cross_bear and rsi_not_oversold and rsi_bearish_zone:
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
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            # Exit long if 12h trend turns bearish OR HMA cross reverses
            if (hma_12h_slope_bear and price_below_hma_12h) or hma_cross_bear:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 12h trend turns bullish OR HMA cross reverses
            if (hma_12h_slope_bull and price_above_hma_12h) or hma_cross_bull:
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
                # Flip position
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