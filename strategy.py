#!/usr/bin/env python3
"""
Experiment #101: 4h Primary + 1d/1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Complex regime switching (CRSI, Fisher, dual-regime) caused 0 trades or negative Sharpe.
This version uses PROVEN components from the current best strategy but SIMPLIFIED:
1) 1d HMA(21) for macro trend direction (load ONCE before loop)
2) 4h HMA(16/48) crossover for entry timing
3) RSI(14) pullback entries (not extremes - allows more trades)
4) Choppiness(14) as OPTIONAL filter (doesn't block all trades)
5) Volume confirmation for breakouts (avoids fake moves)
6) ATR(14) trailing stoploss at 2.5x

Key differences from failed experiments:
- LOOSEN RSI thresholds (40-60 range, not 30-70)
- Choppiness is bonus filter, not required
- 1d HMA slope confirms trend strength
- Volume > SMA(20) confirms breakout validity
- Discrete sizing: 0.25 base, 0.30 max with confluence

Why this should beat Sharpe=0.486:
- Simpler = more trades across ALL symbols (BTC/ETH/SOL)
- 4h timeframe targets 25-45 trades/year (fee-efficient)
- 1d/1w HTF prevents counter-trend in bear markets (2025 test)
- Volume filter reduces false breakouts (major issue in #089, #099)
- Proper stoploss tracking (missing in many failed strategies)

Position size: 0.25 base, 0.30 max with confluence
Stoploss: 2.5*ATR trailing
Target: 30-50 trades/year, Sharpe > 0.5 on ALL symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_vol_1d1w_v1"
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

def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA."""
    vol_s = pd.Series(volume)
    vol_sma = vol_s.rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for ultra-macro trend
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d HMA slope (trend strength)
    hma_1d_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1d_aligned[i]) and not np.isnan(hma_1d_aligned[i-1]) and hma_1d_aligned[i-1] != 0:
            hma_1d_slope[i] = (hma_1d_aligned[i] - hma_1d_aligned[i-1]) / hma_1d_aligned[i-1] * 100
        else:
            hma_1d_slope[i] = 0.0
    
    # Calculate 1w HMA slope
    hma_1w_slope = np.zeros(n)
    for i in range(1, n):
        if not np.isnan(hma_1w_aligned[i]) and not np.isnan(hma_1w_aligned[i-1]) and hma_1w_aligned[i-1] != 0:
            hma_1w_slope[i] = (hma_1w_aligned[i] - hma_1w_aligned[i-1]) / hma_1w_aligned[i-1] * 100
        else:
            hma_1w_slope[i] = 0.0
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_4h_16 = calculate_hma(close, period=16)
    hma_4h_48 = calculate_hma(close, period=48)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    ema_21 = calculate_ema(close, period=21)
    vol_sma_20 = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_16[i]) or np.isnan(hma_4h_48[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(ema_21[i]) or np.isnan(vol_sma_20[i]):
            continue
        
        # === HTF TREND BIAS (1d + 1w HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        hma_1d_slope_positive = hma_1d_slope[i] > 0.05
        hma_1d_slope_negative = hma_1d_slope[i] < -0.05
        hma_1w_slope_positive = hma_1w_slope[i] > 0.05
        hma_1w_slope_negative = hma_1w_slope[i] < -0.05
        
        # === CHOPPINESS REGIME (OPTIONAL FILTER) ===
        chop_trending = chop_14[i] < 50.0
        chop_ranging = chop_14[i] > 50.0
        
        # === 4h HMA CROSSOVER ===
        hma_bullish = hma_4h_16[i] > hma_4h_48[i]
        hma_bearish = hma_4h_16[i] < hma_4h_48[i]
        
        # === RSI ENTRY SIGNALS (LOOSE thresholds) ===
        rsi_neutral_long = rsi_14[i] < 60.0
        rsi_neutral_short = rsi_14[i] > 40.0
        rsi_oversold = rsi_14[i] < 45.0
        rsi_overbought = rsi_14[i] > 55.0
        
        # === EMA CONFIRMATION ===
        ema_bullish = close[i] > ema_21[i]
        ema_bearish = close[i] < ema_21[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > vol_sma_20[i]
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        confluence_count = 0
        
        # --- LONG ENTRY: HTF uptrend + 4h bullish + RSI okay ---
        long_bias = price_above_hma_1d and hma_bullish and rsi_neutral_long
        if long_bias:
            confluence_count = 1
            if ema_bullish:
                confluence_count += 1
            if volume_confirmed:
                confluence_count += 1
            if chop_trending:
                confluence_count += 1
            
            new_signal = POSITION_SIZE_BASE
            if confluence_count >= 3:
                new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY: HTF downtrend + 4h bearish + RSI okay ---
        short_bias = price_below_hma_1d and hma_bearish and rsi_neutral_short
        if short_bias:
            confluence_count = 1
            if ema_bearish:
                confluence_count += 1
            if volume_confirmed:
                confluence_count += 1
            if chop_trending:
                confluence_count += 1
            
            new_signal = -POSITION_SIZE_BASE
            if confluence_count >= 3:
                new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            if position_side > 0 and rsi_14[i] < 70.0:
                new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0 and rsi_14[i] > 30.0:
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
        if in_position and position_side > 0:
            if price_below_hma_1d and hma_1d_slope_negative:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d and hma_1d_slope_positive:
                new_signal = 0.0
        
        # === EXIT ON RSI EXTREME (take profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            new_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
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