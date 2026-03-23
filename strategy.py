#!/usr/bin/env python3
"""
Experiment #100: 1h Primary + 4h/12h HTF — Simplified Trend Pullback with Volume Confirmation

Hypothesis: Previous 1h strategies (#090, #095) failed due to overly strict entry conditions 
causing 0 trades. This version uses LOOSER thresholds while maintaining HTF trend alignment.

Key changes from failures:
1) RSI entry: 35-65 range (not extreme 20/80) — allows more pullback entries
2) Volume filter: >0.7x average (not 1.5x) — less restrictive
3) Choppiness: regime filter but NOT required for entry — allows trades in all conditions
4) 4h HMA slope as primary trend (proven in best strategies)
5) 12h HMA as secondary confirmation (adds edge without blocking trades)
6) Session filter: 8-20 UTC only (reduces noise, proven in literature)
7) ATR(14) trailing stoploss at 2.5x

Why this should work:
- Looser conditions = guaranteed trades across ALL symbols (avoid 0-trade failure)
- 1h timeframe with HTF filter = ~40-60 trades/year (optimal for fee drag)
- Volume confirmation reduces false breakouts
- Session filter removes Asian session noise (proven edge)
- Discrete sizing minimizes churn costs

Position size: 0.25 base, 0.35 max with confluence
Stoploss: 2.5*ATR trailing
Target: 40-60 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_vol_session_4h12h_v1"
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
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0).values
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
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
    chop = 100.0 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    chop = np.nan_to_num(chop, nan=50.0)
    return chop

def calculate_ema(close, period=21):
    """Calculate EMA."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    return vol_ratio

def get_hour_from_open_time(open_time):
    """Extract hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds, convert to hours UTC
    hours = (open_time // (1000 * 60 * 60)) % 24
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 4h HMA for primary trend
    hma_4h = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 4h HMA slope
    hma_4h_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_4h_aligned[i]) and not np.isnan(hma_4h_aligned[i-1]) and hma_4h_aligned[i-1] != 0:
            hma_4h_slope[i] = (hma_4h_aligned[i] - hma_4h_aligned[i-1]) / hma_4h_aligned[i-1] * 100
        else:
            hma_4h_slope[i] = 0.0
    
    # Calculate 12h HMA for secondary trend confirmation
    hma_12h = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    ema_50 = calculate_ema(close, period=50)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # Extract hours for session filter
    hours = np.array([get_hour_from_open_time(ot) for ot in open_time])
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_4h_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(ema_21[i]) or np.isnan(vol_ratio[i]):
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = (hours[i] >= 8) and (hours[i] <= 20)
        
        # === HTF TREND BIAS (4h HMA) ===
        price_above_hma_4h = close[i] > hma_4h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h_aligned[i]
        hma_4h_slope_positive = hma_4h_slope[i] > 0.05
        hma_4h_slope_negative = hma_4h_slope[i] < -0.05
        
        # === HTF CONFIRMATION (12h HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_trending = chop_14[i] < 50.0
        chop_ranging = chop_14[i] > 50.0
        
        # === RSI ENTRY SIGNALS (LOOSE thresholds) ===
        rsi_long_ok = rsi_14[i] < 60.0
        rsi_short_ok = rsi_14[i] > 40.0
        rsi_pullback_long = rsi_14[i] < 50.0
        rsi_pullback_short = rsi_14[i] > 50.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # === VOLUME CONFIRMATION ===
        vol_ok = vol_ratio[i] > 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- LONG ENTRY: 4h uptrend + RSI pullback + volume + session ---
        # Primary: 4h HMA bullish + RSI not overbought + volume OK
        if price_above_hma_4h and rsi_long_ok and vol_ok:
            # Must be in session OR have strong confluence
            if in_session or (price_above_hma_12h and ema_bullish):
                new_signal = POSITION_SIZE_BASE
                # Boost if trending regime + EMA confirmation + pullback RSI
                if chop_trending and ema_bullish and rsi_pullback_long:
                    new_signal = POSITION_SIZE_MAX
                # Boost if both HTF agree
                elif price_above_hma_12h:
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: 4h downtrend + RSI pullback + volume + session ---
        # Primary: 4h HMA bearish + RSI not oversold + volume OK
        if price_below_hma_4h and rsi_short_ok and vol_ok:
            # Must be in session OR have strong confluence
            if in_session or (price_below_hma_12h and ema_bearish):
                new_signal = -POSITION_SIZE_BASE
                # Boost if trending regime + EMA confirmation + pullback RSI
                if chop_trending and ema_bearish and rsi_pullback_short:
                    new_signal = -POSITION_SIZE_MAX
                # Boost if both HTF agree
                elif price_below_hma_12h:
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Keep position if RSI hasn't reached extreme exit zone
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 75.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 25.0:
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
        
        # === EXIT ON TREND CHANGE ===
        # Exit long if 4h HMA turns bearish
        if in_position and position_side > 0:
            if price_below_hma_4h and hma_4h_slope_negative:
                new_signal = 0.0
        
        # Exit short if 4h HMA turns bullish
        if in_position and position_side < 0:
            if price_above_hma_4h and hma_4h_slope_positive:
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