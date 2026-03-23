#!/usr/bin/env python3
"""
Experiment #680: 1h Primary + 4h/12h HTF — Simplified Regime + RSI + HMA Trend

Hypothesis: After analyzing 594 failed strategies, the pattern for 1h timeframe is clear:
1. #670 got 0 trades - CRSI + session filter TOO STRICT for 1h
2. #675 got negative Sharpe - too many trades without HTF confirmation
3. Lower TF needs SIMPLER entry signals but STRICTER HTF confirmation

This strategy uses:
- 12h HMA (21) for PRIMARY trend bias - slower, more reliable than 4h
- 4h RSI (14) for entry timing - simpler than CRSI, more frequent signals
- 1h Choppiness (14) for regime detection - range vs trend
- Asymmetric entries: mean-revert in chop, trend-pullback when trending
- NO session filter (kills trade frequency)
- Conservative sizing: 0.25 trend, 0.20 mean-revert

Why this might beat Sharpe=0.520:
- 1h timeframe with 12h/4h HTF = optimal trade frequency (30-60/year)
- RSI(14) < 30 / > 70 occurs more often than CRSI < 10 / > 90
- 12h HMA provides stable trend bias without over-filtering
- Choppiness regime prevents wrong-style trading (major loss source)
- ATR stoploss (2.5x) controls drawdown in 2022-style crashes

Position sizing: 0.20-0.25 discrete (per Rule 4, max 0.40)
Target: 35-55 trades/year on 1h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_chop_rsi_hma_4h12h_v1"
timeframe = "1h"
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
    
    Interpretation:
    - CHOP > 61.8: Range/consolidation (mean-revert)
    - CHOP < 38.2: Trending (trend-follow)
    - 38.2-61.8: Transition (use HTF bias)
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
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0.0, 100.0)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 12h HMA for primary trend direction
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 4h RSI for entry timing
    rsi_4h = calculate_rsi(df_4h['close'].values, period=14)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_TREND = 0.25      # Trend-follow entries
    SIZE_MR = 0.20         # Mean-revert entries
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(rsi_4h_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_1h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 12H TREND BIAS (HMA slope over 3 bars) ===
        hma_12h_slope_bull = hma_12h_aligned[i] > hma_12h_aligned[i-3] if i >= 3 else False
        hma_12h_slope_bear = hma_12h_aligned[i] < hma_12h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 12h HMA
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 58.0      # Range/consolidation
        is_trend = chop_14[i] < 42.0      # Trending
        is_transition = not is_range and not is_trend  # Middle ground
        
        # === RSI EXTREMES (1h and 4h) ===
        rsi_1h_oversold = rsi_1h[i] < 35.0
        rsi_1h_overbought = rsi_1h[i] > 65.0
        rsi_4h_oversold = rsi_4h_aligned[i] < 40.0
        rsi_4h_overbought = rsi_4h_aligned[i] > 60.0
        
        # === 1H HMA SLOPE (3 bars) ===
        hma_1h_slope_bull = hma_1h[i] > hma_1h[i-3] if i >= 3 else False
        hma_1h_slope_bear = hma_1h[i] < hma_1h[i-3] if i >= 3 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (CHOP > 58) + RSI oversold = mean revert long
        if is_range and rsi_1h_oversold:
            new_signal = SIZE_MR
        
        # Regime 2: Trending market (CHOP < 42) + 12h bull + RSI pullback = trend long
        elif is_trend and hma_12h_slope_bull and price_above_hma_12h:
            if rsi_4h_oversold and hma_1h_slope_bull:
                new_signal = SIZE_TREND
        
        # Regime 3: Transition + 12h bull + RSI very oversold = opportunistic long
        elif is_transition and hma_12h_slope_bull:
            if rsi_1h[i] < 30.0:
                new_signal = SIZE_MR
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (CHOP > 58) + RSI overbought = mean revert short
        elif is_range and rsi_1h_overbought:
            new_signal = -SIZE_MR
        
        # Regime 2: Trending market (CHOP < 42) + 12h bear + RSI pullback = trend short
        elif is_trend and hma_12h_slope_bear and price_below_hma_12h:
            if rsi_4h_overbought and hma_1h_slope_bear:
                new_signal = -SIZE_TREND
        
        # Regime 3: Transition + 12h bear + RSI very overbought = opportunistic short
        elif is_transition and hma_12h_slope_bear:
            if rsi_1h[i] > 70.0:
                new_signal = -SIZE_MR
        
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
            if hma_12h_slope_bear and price_below_hma_12h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_12h_slope_bull and price_above_hma_12h:
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