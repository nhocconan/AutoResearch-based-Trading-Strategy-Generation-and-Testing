#!/usr/bin/env python3
"""
Experiment #148: 30m Primary + 4h/1d HTF — Simplified Regime + RSI Pullback

Hypothesis: Previous 30m/1h strategies failed due to either (a) too many trades 
causing fee drag, or (b) too many conflicting filters causing 0 trades.

This strategy uses SIMPLIFIED but PROVEN components:
1) 1d HMA(21) for macro trend bias — ONLY trade with HTF trend direction
2) 4h Choppiness Index(14) for regime — >55 = range, <45 = trend
3) 30m RSI(7) pullback entries — enter on dips in uptrend, rallies in downtrend
4) Volume confirmation — volume > 0.8x 20-bar average
5) Session filter — only trade 8-20 UTC (high liquidity hours)
6) ATR(14) trailing stop at 2.5x — protects capital

Why this should work on 30m:
- 1d HMA ensures we only trade with macro trend (reduces whipsaws)
- Choppiness adapts entry logic to market regime
- RSI(7) pullback is proven for trend continuation entries
- Session filter reduces noise during low-liquidity hours
- 30m should produce 40-80 trades/year with these filters

Position size: 0.20 base, 0.25 with volume confluence (smaller for lower TF)
Stoploss: 2.5*ATR trailing
Target: 40-80 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_regime_rsi_pullback_4h1d_session_v1"
timeframe = "30m"
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

def calculate_rsi(close, period=7):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50).values

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures market choppy vs trending.
    CHOP > 61.8 = range/choppy, CHOP < 38.2 = trending
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
    price_range = np.maximum(price_range, 1e-10)
    
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_volume_avg(volume, period=20):
    """Calculate rolling average volume."""
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_hour_from_open_time(open_time_array):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to hours UTC
    hours = (open_time_array // (1000 * 60 * 60)) % 24
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
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h Choppiness for regime
    chop_4h_raw = calculate_choppiness(
        df_4h['high'].values, 
        df_4h['low'].values, 
        df_4h['close'].values, 
        period=14
    )
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_7 = calculate_rsi(close, period=7)
    vol_avg_20 = calculate_volume_avg(volume, period=20)
    
    # Extract hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.20
    POSITION_SIZE_MAX = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_4h_aligned[i]):
            continue
        if np.isnan(rsi_7[i]) or np.isnan(vol_avg_20[i]) or vol_avg_20[i] == 0:
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        is_choppy = chop_4h_aligned[i] > 55.0
        is_trending = chop_4h_aligned[i] < 45.0
        
        # === VOLUME ===
        volume_ratio = volume[i] / (vol_avg_20[i] + 1e-10)
        volume_confirmed = volume_ratio > 0.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # Only trade during session hours
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
            continue
        
        # --- TREND REGIME: RSI Pullback Entry ---
        if is_trending:
            # Long: price above 1d HMA + RSI(7) pullback to 35-45
            if price_above_hma_1d:
                if 35.0 <= rsi_7[i] <= 50.0:
                    if volume_confirmed:
                        new_signal = POSITION_SIZE_BASE
                        if volume_ratio > 1.5:
                            new_signal = POSITION_SIZE_MAX
            
            # Short: price below 1d HMA + RSI(7) rally to 50-65
            if price_below_hma_1d:
                if 50.0 <= rsi_7[i] <= 65.0:
                    if volume_confirmed:
                        new_signal = -POSITION_SIZE_BASE
                        if volume_ratio > 1.5:
                            new_signal = -POSITION_SIZE_MAX
        
        # --- RANGE REGIME: RSI Extreme Mean Reversion ---
        if is_choppy:
            # Long: RSI < 25 (oversold) + price above 1d HMA bias
            if rsi_7[i] < 25.0:
                if price_above_hma_1d or not price_below_hma_1d:
                    if volume_confirmed:
                        new_signal = POSITION_SIZE_BASE
            
            # Short: RSI > 75 (overbought) + price below 1d HMA bias
            if rsi_7[i] > 75.0:
                if price_below_hma_1d or not price_above_hma_1d:
                    if volume_confirmed:
                        new_signal = -POSITION_SIZE_BASE
        
        # === HOLD POSITION LOGIC ===
        # Hold if no new signal and position still valid
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if RSI not overbought
                if rsi_7[i] < 70.0:
                    new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if RSI not oversold
                if rsi_7[i] > 30.0:
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
            if price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_7[i] > 75.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_7[i] < 25.0:
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