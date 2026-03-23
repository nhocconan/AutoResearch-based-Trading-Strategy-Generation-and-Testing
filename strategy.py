#!/usr/bin/env python3
"""
Experiment #659: 4h Primary + 1d HTF — Vol-Spike Mean Reversion + Choppiness Regime

Hypothesis: After analyzing 577 failed strategies and current #651 (Sharpe=0.222):
1. #651's CRSI thresholds (<15, >85) are TOO EXTREME = missed many valid reversals
2. Vol-spike mean reversion has proven edge in literature (vol crush after panic)
3. Combining vol-spike detection with Choppiness regime = better entry timing
4. Need MORE trades (current may be too conservative) while maintaining quality

This strategy uses:
- ATR Ratio (ATR7/ATR30) > 1.8 to detect vol spikes (panic conditions)
- Choppiness Index (14) for regime: >55 range, <45 trend
- RSI(7) extremes for entry timing (softer than CRSI: <25 long, >75 short)
- 1d HMA(21) slope for major trend bias
- Bollinger Band position for additional mean-reversion confirmation

Why this might beat Sharpe=0.520:
- Vol-spike entries have higher win rate (panic reversals)
- Softer RSI thresholds = more trades (target 35-50/year on 4h)
- BB position filter adds confluence without over-filtering
- 1d HMA keeps us on right side of major moves

Position sizing: 0.28 discrete (slightly lower than 0.30 for safety)
Target: 35-50 trades/year on 4h
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_volspike_chop_rsi_bb_1d_v1"
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
    CHOP > 61.8: Range | CHOP < 38.2: Trend
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    
    return upper, lower, sma

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for vol spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        atr_ratio = atr_short / (atr_long + 1e-10)
    
    return atr_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for primary trend direction
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    chop_14 = calculate_choppiness(high, low, close, 14)
    rsi_7 = calculate_rsi(close, 7)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(rsi_7[i]) or np.isnan(atr_ratio[i]):
            continue
        if atr_14[i] == 0 or np.isnan(bb_upper[i]):
            continue
        
        # === 1D TREND BIAS (HMA slope over 5 bars) ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-5] if i >= 5 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-5] if i >= 5 else False
        
        # Price relative to 1d HMA
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range = chop_14[i] > 55.0
        is_trend = chop_14[i] < 45.0
        
        # === VOLATILITY SPIKE DETECTION ===
        vol_spike = atr_ratio[i] > 1.7  # ATR7 > 1.7x ATR30 = elevated vol
        
        # === RSI EXTREMES (softer than CRSI for more trades) ===
        rsi_oversold = rsi_7[i] < 30.0
        rsi_overbought = rsi_7[i] > 70.0
        
        # === BOLLINGER BAND POSITION ===
        bb_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i] + 1e-10)
        bb_low = bb_position < 0.15  # Near lower band
        bb_high = bb_position > 0.85  # Near upper band
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Scenario 1: Range market + Vol spike + RSI oversold + BB low (mean revert)
        if is_range and vol_spike and rsi_oversold and bb_low:
            new_signal = POSITION_SIZE
        
        # Scenario 2: Range market + RSI very oversold (<20) + BB low
        elif is_range and rsi_7[i] < 20.0 and bb_low:
            new_signal = POSITION_SIZE
        
        # Scenario 3: Trending bull + pullback to BB mid + RSI moderate oversold
        elif is_trend and hma_1d_slope_bull and price_above_hma_1d:
            if bb_position < 0.40 and rsi_7[i] < 40.0:
                new_signal = POSITION_SIZE
        
        # Scenario 4: Vol spike + extreme RSI (<15) regardless of regime (panic reversal)
        if vol_spike and rsi_7[i] < 15.0:
            new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY ---
        # Scenario 1: Range market + Vol spike + RSI overbought + BB high (mean revert)
        if is_range and vol_spike and rsi_overbought and bb_high:
            new_signal = -POSITION_SIZE
        
        # Scenario 2: Range market + RSI very overbought (>80) + BB high
        elif is_range and rsi_7[i] > 80.0 and bb_high:
            new_signal = -POSITION_SIZE
        
        # Scenario 3: Trending bear + pullback to BB mid + RSI moderate overbought
        elif is_trend and hma_1d_slope_bear and price_below_hma_1d:
            if bb_position > 0.60 and rsi_7[i] > 60.0:
                new_signal = -POSITION_SIZE
        
        # Scenario 4: Vol spike + extreme RSI (>85) regardless of regime (panic reversal)
        if vol_spike and rsi_7[i] > 85.0:
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
            if hma_1d_slope_bear and price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if hma_1d_slope_bull and price_above_hma_1d:
                new_signal = 0.0
        
        # === TAKE PROFIT (reduce to half at 2R) ===
        if in_position and position_side > 0:
            if close[i] > entry_price + 2.0 * atr_14[i]:
                if new_signal == POSITION_SIZE:
                    new_signal = POSITION_SIZE / 2.0
        
        if in_position and position_side < 0:
            if close[i] < entry_price - 2.0 * atr_14[i]:
                if new_signal == -POSITION_SIZE:
                    new_signal = -POSITION_SIZE / 2.0
        
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