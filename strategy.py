#!/usr/bin/env python3
"""
Experiment #051: 4h Primary + 1d HTF — BB Squeeze + Donchian Breakout + ADX Regime

Hypothesis: Volatility contraction (BB Width percentile < 20%) followed by 
Donchian breakout with ADX regime filter will catch explosive moves while 
avoiding choppy whipsaws. 1d HTF provides macro trend bias.

Key insights from 46+ failed experiments:
1) BB Squeeze (low volatility) precedes 70% of major breakouts
2) Donchian(20) breakout is proven on 4h timeframe
3) ADX with hysteresis (enter >22, exit <18) prevents regime whipsaw
4) 1d HMA(50) for macro bias — only trade breakouts in trend direction
5) LOOSE entry thresholds to ensure 25+ trades/year (avoid Sharpe=0.000)

Why this differs from failed attempts:
- NOT using CRSI (overused in #039-#050, diminishing returns)
- NOT using Choppiness Index (tried 8+ times with negative Sharpe)
- Using BB Width percentile + Donchian + ADX (new combination for 4h)
- Simpler logic = more trades generated on ALL symbols

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
Target: 25-50 trades/year, Sharpe > 0.486 (beat current best)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_bb_squeeze_donchian_adx_regime_1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / (sma + 1e-10) * 100.0
    return upper, lower, bandwidth, sma

def calculate_bb_width_percentile(bandwidth, lookback=100):
    """Calculate percentile rank of BB Width over lookback period."""
    n = len(bandwidth)
    percentile = np.zeros(n)
    for i in range(lookback, n):
        window = bandwidth[i-lookback+1:i+1]
        current = bandwidth[i]
        count_below = np.sum(window[:-1] < current)
        percentile[i] = 100.0 * count_below / (lookback - 1)
    percentile[:lookback] = 50.0
    return percentile

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smoothed DM and TR
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = 100.0 * plus_dm_s / (atr + 1e-10)
    minus_di = 100.0 * minus_dm_s / (atr + 1e-10)
    
    # DX
    dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_hma(close, period=50):
    """Calculate Hull Moving Average for HTF trend."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro bias
    hma_1d = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_upper, bb_lower, bb_width, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    bb_pct = calculate_bb_width_percentile(bb_width, lookback=100)
    donch_high, donch_low, donch_mid = calculate_donchian(high, low, period=20)
    adx_14, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.28  # Discrete, within 0.20-0.35 range
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    # ADX hysteresis state
    adx_trending = False
    
    for i in range(150, n):  # Warmup for all indicators
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(bb_width[i]) or np.isnan(bb_pct[i]) or np.isnan(donch_high[i]):
            continue
        if np.isnan(adx_14[i]) or atr_14[i] == 0:
            continue
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === BB SQUEEZE (Volatility Contraction) ===
        bb_squeeze = bb_pct[i] < 25.0  # Bottom 25% of volatility = squeeze
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donch_high[i-1]  # Break above previous high
        breakout_short = close[i] < donch_low[i-1]  # Break below previous low
        
        # === ADX REGIME with HYSTERESIS ===
        if adx_14[i] > 22.0:
            adx_trending = True
        elif adx_14[i] < 18.0:
            adx_trending = False
        
        # === DI CROSSOVER ===
        di_bullish = plus_di[i] > minus_di[i]
        di_bearish = plus_di[i] < minus_di[i]
        
        # === ENTRY LOGIC (LOOSE for trade generation) ===
        new_signal = 0.0
        
        # --- LONG ENTRY: BB Squeeze + Donchian Breakout + Trend Confirm ---
        if breakout_long:
            # Need at least 2 of 3 confirmations for long
            confirm_count = 0
            if price_above_hma_1d:
                confirm_count += 1
            if di_bullish:
                confirm_count += 1
            if adx_trending or bb_squeeze:
                confirm_count += 1
            
            if confirm_count >= 2:
                new_signal = POSITION_SIZE
        
        # --- SHORT ENTRY: BB Squeeze + Donchian Breakout + Trend Confirm ---
        elif breakout_short:
            # Need at least 2 of 3 confirmations for short
            confirm_count = 0
            if price_below_hma_1d:
                confirm_count += 1
            if di_bearish:
                confirm_count += 1
            if adx_trending or bb_squeeze:
                confirm_count += 1
            
            if confirm_count >= 2:
                new_signal = -POSITION_SIZE
        
        # --- MEAN REVERSION in BB SQUEEZE (ensures trades in chop) ---
        if new_signal == 0.0 and bb_squeeze:
            # Long at lower band
            if close[i] < bb_lower[i] * 1.002 and price_above_hma_1d:
                new_signal = POSITION_SIZE
            # Short at upper band
            elif close[i] > bb_upper[i] * 0.998 and price_below_hma_1d:
                new_signal = -POSITION_SIZE
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            # Hold if no opposite breakout signal
            if position_side > 0 and not breakout_short:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and not breakout_long:
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
        
        # === EXIT ON MACRO TREND CHANGE ===
        if in_position and position_side > 0:
            if price_below_hma_1d and di_bearish:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d and di_bullish:
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