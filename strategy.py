#!/usr/bin/env python3
"""
Experiment #145: 1h Primary + 4h/1d HTF — Pullback Strategy with Session Filter

Hypothesis: Previous 1h strategies (#135, #140) failed with 0 trades due to overly 
strict entry conditions. This strategy uses PROVEN pullback logic with LOOSER 
thresholds to ensure trade generation while controlling frequency via session filter.

Key Components:
1) 4h HMA(21) for trend bias — only trade in HTF trend direction
2) 1d HMA(21) for macro filter — adds confluence
3) 1h RSI(14) pullback entries — long on dip to 35-45, short on rally to 55-65
4) Session filter: 8-20 UTC only — reduces trades by ~50%, focuses on liquid hours
5) Volume confirmation: >0.8x avg (not strict 1.3x) — ensures minimum liquidity
6) ATR(14) trailing stop at 2.5x — protects capital
7) Position size: 0.25 base (smaller for 1h TF to reduce fee impact)

Why this should work:
- Pullback entries have higher win rate than breakouts in range/bear markets
- Session filter naturally limits trades to 30-60/year on 1h
- Looser RSI thresholds (35-45/55-65 vs 20/80) ensure trade generation
- 4h HMA proven in best strategy (Sharpe=0.486)
- Simpler than failed regime-switching strategies

Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h1d_session_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_session_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to seconds then to datetime
    timestamps = open_time / 1000.0
    hours = pd.to_datetime(timestamps, unit='s').hour
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h HMA for trend bias
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1d HMA for macro filter
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    session_hours = get_session_hour(open_time)
    
    signals = np.zeros(n)
    POSITION_SIZE = 0.25
    STOPLOSS_MULT = 2.5
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        current_hour = session_hours[i]
        in_session = (current_hour >= 8) and (current_hour <= 20)
        
        # Only trade during session hours
        if not in_session:
            # If in position, hold it; otherwise stay flat
            if in_position:
                signals[i] = signals[i-1] if i > 0 else 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === MACRO FILTER (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_ok = volume_ratio > 0.8
        
        # === RSI PULLBACK LOGIC ===
        rsi_value = rsi_14[i]
        
        # Long setup: 4h bullish + RSI pullback to 35-45
        long_setup = price_above_hma_4h and (rsi_value >= 35.0) and (rsi_value <= 45.0)
        
        # Short setup: 4h bearish + RSI rally to 55-65
        short_setup = price_below_hma_4h and (rsi_value >= 55.0) and (rsi_value <= 65.0)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Long entry
        if long_setup and volume_ok:
            # Add 1d HMA confluence for stronger signal
            if price_above_hma_1d:
                new_signal = POSITION_SIZE
            else:
                # Still enter but smaller size if 1d neutral
                new_signal = POSITION_SIZE * 0.8
        
        # Short entry
        if short_setup and volume_ok:
            # Add 1d HMA confluence for stronger signal
            if price_below_hma_1d:
                new_signal = -POSITION_SIZE
            else:
                # Still enter but smaller size if 1d neutral
                new_signal = -POSITION_SIZE * 0.8
        
        # === HOLD POSITION LOGIC ===
        # Hold if in position and no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h trend still bullish
                if price_above_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h trend still bearish
                if price_below_hma_4h:
                    new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - STOPLOSS_MULT * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + STOPLOSS_MULT * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_4h:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_value > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_value < 30.0:
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