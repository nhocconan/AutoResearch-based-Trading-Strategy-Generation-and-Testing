#!/usr/bin/env python3
"""
Experiment #114: 4h Primary + 12h/1d HTF — Donchian Breakout with RSI + Volume Filter

Hypothesis: Previous complex regime-switching strategies failed due to lag and overfitting.
Pure trend-following with Donchian breakouts worked on 4h (Sharpe=0.486 baseline). This 
improves it with:

1) 12h HMA(21) for macro trend bias — only trade breakouts in trend direction
2) 4h Donchian(20) breakout — price breaks 20-bar high/low for entry
3) 4h RSI(14) filter — avoid overbought (>70) longs and oversold (<30) shorts
4) Volume confirmation — breakout volume > 1.5x 20-bar avg (filters false breakouts)
5) ATR(14) trailing stop at 2.5x — locks in profits, limits drawdown
6) Simple exit: opposite Donchian break OR 12h trend reversal

Why this should work:
- Donchian breakouts are proven trend-following (Turtle Trading)
- 12h HMA filter prevents counter-trend trades in bear markets (2022 crash)
- RSI filter reduces chasing overextended moves
- Volume filter reduces whipsaws in low-liquidity breakouts
- 4h naturally produces 25-50 trades/year (low fee drag)
- Simpler logic = more robust across BTC/ETH/SOL

Position size: 0.25 base, 0.30 max with volume confluence
Stoploss: 2.5*ATR trailing
Target: 25-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_rsi_vol_12h_v1"
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-bar high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    gain = np.maximum(delta, 0)
    loss = -np.minimum(delta, 0)
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA for macro trend
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 12h HMA slope (trend strength)
    hma_12h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_12h_aligned[i]) and not np.isnan(hma_12h_aligned[i-1]) and hma_12h_aligned[i-1] != 0:
            hma_12h_slope[i] = (hma_12h_aligned[i] - hma_12h_aligned[i-1]) / hma_12h_aligned[i-1] * 100
        else:
            hma_12h_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    hma_4h_21 = calculate_hma(close, period=21)
    hma_4h_50 = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_12h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_21[i]) or np.isnan(hma_4h_50[i]):
            continue
        
        # === HTF TREND BIAS (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        hma_slope_positive = hma_12h_slope[i] > 0.3
        hma_slope_negative = hma_12h_slope[i] < -0.3
        hma_slope_flat = abs(hma_12h_slope[i]) <= 0.3
        
        # === 4h TREND FILTER ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === DONCHIAN BREAKOUT ===
        prev_high = donchian_upper[i-1] if i > 0 else donchian_upper[i]
        prev_low = donchian_lower[i-1] if i > 0 else donchian_lower[i]
        
        breakout_long = close[i] > prev_high
        breakout_short = close[i] < prev_low
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.5
        volume_strong = volume_ratio > 2.0
        
        # === RSI FILTER ===
        rsi_not_overbought = rsi_14[i] < 70.0
        rsi_not_oversold = rsi_14[i] > 30.0
        rsi_neutral = (rsi_14[i] >= 35.0) and (rsi_14[i] <= 65.0)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        # Require: 12h trend up/flat + 4h trend up + Donchian breakout + RSI not overbought + volume
        if price_above_hma_12h or hma_slope_flat:
            if hma_4h_bullish and breakout_long and rsi_not_overbought:
                if volume_confirmed:
                    new_signal = POSITION_SIZE_BASE
                    if volume_strong and hma_slope_positive and rsi_neutral:
                        new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        # Require: 12h trend down/flat + 4h trend down + Donchian breakout + RSI not oversold + volume
        if price_below_hma_12h or hma_slope_flat:
            if hma_4h_bearish and breakout_short and rsi_not_oversold:
                if volume_confirmed:
                    new_signal = -POSITION_SIZE_BASE
                    if volume_strong and hma_slope_negative and rsi_neutral:
                        new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold long if still above Donchian mid and 12h trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if close[i] > donchian_mid[i] and price_above_hma_12h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                if close[i] < donchian_mid[i] and price_below_hma_12h:
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
        if in_position and position_side > 0:
            if price_below_hma_12h and hma_slope_negative:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_short:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_12h and hma_slope_positive:
                new_signal = 0.0
            # Exit on opposite Donchian break
            if breakout_long:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 25.0:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals