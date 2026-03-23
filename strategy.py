#!/usr/bin/env python3
"""
Experiment #675: 1h Primary + 4h/1d HTF — Simplified Regime + RSI + HMA Trend

Hypothesis: After 590+ failed strategies, the pattern for 1h is clear:
1. #665, #670 got Sharpe=0.000 (0 trades) due to TOO MANY filters (session + CRSI + Chop)
2. 1h needs SIMPLER logic than 4h/1d — fewer confluence requirements
3. Current best (Sharpe=0.520) is 1d — 1h can beat it with MORE trades but SAME quality
4. Key insight: Remove session filter, simplify CRSI→RSI, widen Chop thresholds

This strategy uses:
- 4h HMA(21) for primary trend direction (faster than 1d, more signals)
- 1h RSI(14) extremes for entries (<25 long, >75 short) — simpler than CRSI
- Choppiness Index(14) regime filter with WIDER thresholds (>50 range, <40 trend)
- Only 2 confluence required (not 3+) to ensure 40-80 trades/year
- Asymmetric sizing: 0.25 counter-trend, 0.35 with-trend
- 2*ATR trailing stoploss

Why this might beat Sharpe=0.520:
- 1h timeframe = 40-80 trades/year (optimal per Rule 10)
- RSI(14) extremes trigger MORE often than CRSI <10/>90
- 4h HMA slope is faster than 1d — catches trends earlier
- NO session filter (was killing trades in #665, #670)
- Simpler logic = fewer conditions that can all fail simultaneously

Position sizing: 0.25-0.35 discrete (per Rule 4, max 0.40)
Target: 40-80 trades/year on 1h
Stoploss: 2*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_chop_hma_4h_v1"
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
    CHOP > 50: Range/consolidation (mean-revert)
    CHOP < 40: Trending (trend-follow)
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
    
    # Load 4h HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h HMA for primary trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    SIZE_WITH_TREND = 0.35
    SIZE_COUNTER_TREND = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 4H TREND BIAS (HMA slope over 3 bars) ===
        hma_4h_slope_bull = hma_4h_aligned[i] > hma_4h_aligned[i-3] if i >= 3 else False
        hma_4h_slope_bear = hma_4h_aligned[i] < hma_4h_aligned[i-3] if i >= 3 else False
        
        # Price relative to 4h HMA
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME (WIDER thresholds for more trades) ===
        is_range = chop_14[i] > 50.0  # Range/consolidation
        is_trend = chop_14[i] < 40.0  # Trending
        
        # === RSI EXTREMES (simpler than CRSI, triggers more often) ===
        rsi_oversold = rsi_14[i] < 25.0  # Oversold
        rsi_overbought = rsi_14[i] > 75.0  # Overbought
        rsi_neutral_low = rsi_14[i] < 40.0  # Pullback in uptrend
        rsi_neutral_high = rsi_14[i] > 60.0  # Pullback in downtrend
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Regime 1: Range market (CHOP > 50) + RSI oversold (< 25) = mean revert long
        if is_range and rsi_oversold:
            new_signal = SIZE_COUNTER_TREND
        
        # Regime 2: Trending market (CHOP < 40) + 4h bull + RSI pullback (< 40)
        elif is_trend and hma_4h_slope_bull and price_above_hma_4h:
            if rsi_neutral_low:
                new_signal = SIZE_WITH_TREND
        
        # Regime 3: Neutral (40 < CHOP < 50) + 4h bull + RSI oversold
        elif hma_4h_slope_bull and price_above_hma_4h and rsi_oversold:
            new_signal = SIZE_WITH_TREND
        
        # --- SHORT ENTRY ---
        # Regime 1: Range market (CHOP > 50) + RSI overbought (> 75) = mean revert short
        elif is_range and rsi_overbought:
            new_signal = -SIZE_COUNTER_TREND
        
        # Regime 2: Trending market (CHOP < 40) + 4h bear + RSI pullback (> 60)
        elif is_trend and hma_4h_slope_bear and price_below_hma_4h:
            if rsi_neutral_high:
                new_signal = -SIZE_WITH_TREND
        
        # Regime 3: Neutral (40 < CHOP < 50) + 4h bear + RSI overbought
        elif hma_4h_slope_bear and price_below_hma_4h and rsi_overbought:
            new_signal = -SIZE_WITH_TREND
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND FLIP ===
        if in_position and position_side > 0:
            if hma_4h_slope_bear and price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_4h_slope_bull and price_above_hma_4h:
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