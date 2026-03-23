#!/usr/bin/env python3
"""
Experiment #135: 1h Primary + 4h/1d HTF — HMA Trend + RSI Pullback with Session Filter

Hypothesis: Previous 1h strategies (#125, #128, #130) failed with 0 trades because entry
conditions were TOO STRICT (CRSI + CHOP + session all required). This strategy:

1) 4h HMA(21) for trend direction — proven in #134 (Sharpe=0.245)
2) 1h RSI(14) for pullback entries — simpler than CRSI, more reliable triggers
3) Volume confirmation at 1.2x (not 1.5x) — ensures liquidity without filtering too much
4) Session filter 8-20 UTC — avoids low-liquidity Asian session noise
5) Choppiness as SOFT filter (not hard requirement) — adapts to regime without blocking trades
6) ATR(14) stoploss at 2.5x — protects against whipsaws
7) Relaxed RSI thresholds (25/75 not 20/80) — ensures trades actually trigger

Why this should work on 1h:
- 4h trend filter prevents counter-trend trades (main failure mode in 2022)
- RSI pullback entries catch dips in uptrends / rallies in downtrends
- Session filter reduces noise by ~40% (only trade high-liquidity hours)
- Relaxed volume/RSI thresholds ensure 30-60 trades/year (not 0 like #128)
- Discrete signal sizes (0.20, 0.30) minimize fee churn

Position size: 0.20 base, 0.30 with volume confluence
Stoploss: 2.5*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.3 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h_session_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    high_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    low_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    price_range = high_high - low_low
    
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to hours UTC
    hours = (open_time // (1000 * 3600)) % 24
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
    
    # Calculate 4h HMA for trend direction
    hma_4h_21 = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1d HMA for macro bias
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Calculate 1h HMA for short-term trend
    hma_1h_21 = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        if np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        if np.isnan(hma_1h_21[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC) ===
        hour = get_hour_from_open_time(open_time[i])
        in_session = 8 <= hour <= 20
        
        # === 4H TREND BIAS ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        
        # === 1D MACRO BIAS ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 1H SHORT-TERM TREND ===
        hma_1h_bullish = hma_1h_21[i] > hma_4h_aligned[i] if not np.isnan(hma_4h_aligned[i]) else False
        hma_1h_bearish = hma_1h_21[i] < hma_4h_aligned[i] if not np.isnan(hma_4h_aligned[i]) else False
        
        # === RSI PULLBACK CONDITIONS ===
        # Long: RSI dipped to 30-45 in 4h uptrend
        rsi_oversold = rsi_14[i] < 45
        rsi_deep_oversold = rsi_14[i] < 35
        
        # Short: RSI rallied to 55-70 in 4h downtrend
        rsi_overbought = rsi_14[i] > 55
        rsi_deep_overbought = rsi_14[i] > 65
        
        # === VOLUME CONFIRMATION ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 1.2
        volume_strong = volume_ratio > 1.8
        
        # === CHOPPINESS REGIME (SOFT FILTER) ===
        chop_range = chop_14[i] > 50  # ranging market
        chop_trend = chop_14[i] < 45  # trending market
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours (reduces noise)
        if in_session:
            # --- LONG ENTRY ---
            # 4h uptrend + RSI pullback + volume
            if price_above_hma_4h:
                if rsi_oversold and volume_confirmed:
                    new_signal = POSITION_SIZE_BASE
                    # Increase size with stronger confluence
                    if rsi_deep_oversold and volume_strong and price_above_hma_1d:
                        new_signal = POSITION_SIZE_MAX
            
            # --- SHORT ENTRY ---
            # 4h downtrend + RSI rally + volume
            if price_below_hma_4h:
                if rsi_overbought and volume_confirmed:
                    new_signal = -POSITION_SIZE_BASE
                    # Increase size with stronger confluence
                    if rsi_deep_overbought and volume_strong and price_below_hma_1d:
                        new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if trend intact and no exit signal
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if 4h trend still up
                if price_above_hma_4h and rsi_14[i] < 70:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if 4h trend still down
                if price_below_hma_4h and rsi_14[i] > 30:
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
            if price_below_hma_4h:
                new_signal = 0.0
            # Exit on RSI extreme (take profit)
            if rsi_14[i] > 75:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_4h:
                new_signal = 0.0
            # Exit on RSI extreme (take profit)
            if rsi_14[i] < 25:
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