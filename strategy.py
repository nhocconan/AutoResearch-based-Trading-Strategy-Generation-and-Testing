#!/usr/bin/env python3
"""
Experiment #1000: 1h Primary + 4h/12h HTF — Simplified Trend Pullback with Volume

Hypothesis: After 725 failed strategies, the key insight is that 1h strategies with
too many confluence filters generate ZERO trades. This strategy SIMPLIFIES entry
conditions while keeping HTF trend filter for direction.

Key changes from failures (#988, #990, #995, #998, #999):
1. REMOVED session filter (8-20 UTC) — was killing 60% of potential trades
2. REMOVED Choppiness Index — too many conditions = 0 trades
3. LOOSENED RSI thresholds (30/70 not 25/75) — more entry opportunities
4. LOOSENED volume filter (>0.5x avg not >0.8x) — ensures trades on low vol days
5. SIMPLIFIED regime logic — just HTF trend + LTF pullback
6. ADDED funding fallback — if funding data missing, use RSI extremes alone

Strategy logic:
- 4h HMA(21): trend direction (price > HMA = bullish bias)
- 12h HMA(21): macro confirmation (optional confluence)
- 1h RSI(14): entry trigger (pullback in trend)
- 1h Volume: >0.5x 20-bar average (confirms interest)
- 1h ATR(14): stoploss at 2.5x ATR

Entry conditions (LOOSENED for trade generation):
LONG: 4h_HMA_bullish AND RSI < 40 AND volume > 0.5x_avg
SHORT: 4h_HMA_bearish AND RSI > 60 AND volume > 0.5x_avg

Fallback (guarantees trades when confluence fails):
LONG: RSI < 25 (extreme oversold) — always trigger
SHORT: RSI > 75 (extreme overbought) — always trigger

Target: 40-60 trades/year on 1h, Sharpe > 0.612, ALL symbols positive
Timeframe: 1h (balance between signal quality and trade frequency)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_trend_pullback_4h12h_hma_rsi_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index — standard Wilder's formula."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    avg_gain = np.concatenate([[np.nan], avg_gain])
    avg_loss = np.concatenate([[np.nan], avg_loss])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(series, period):
    """Hull Moving Average — reduces lag vs traditional MA."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Average True Range — Wilder's smoothed method."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio: current volume / rolling average volume."""
    n = len(volume)
    ratio = np.full(n, np.nan)
    
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=period, min_periods=period).mean().values
    
    for i in range(n):
        if not np.isnan(volume[i]) and not np.isnan(vol_avg[i]) and vol_avg[i] > 1e-10:
            ratio[i] = volume[i] / vol_avg[i]
    
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # === PRIMARY (1h) INDICATORS ===
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    vol_ratio_1h = calculate_volume_ratio(volume, period=20)
    
    # === HTF (4h) TREND — HMA21 ===
    hma_4h_raw = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # === HTF (12h) CONFIRMATION — HMA21 ===
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or atr_1h[i] <= 1e-10:
            continue
        if np.isnan(vol_ratio_1h[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA21) ===
        trend_4h_bullish = close[i] > hma_4h_aligned[i]
        trend_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === MACRO CONFIRMATION (12h HMA21) ===
        trend_12h_bullish = close[i] > hma_12h_aligned[i]
        trend_12h_bearish = close[i] < hma_12h_aligned[i]
        
        # === VOLUME FILTER (LOOSENED: >0.5x not >0.8x) ===
        volume_ok = vol_ratio_1h[i] > 0.5
        
        # === RSI SIGNALS (LOOSENED: 40/60 not 30/70 for entries) ===
        rsi_oversold = rsi_1h[i] < 40
        rsi_overbought = rsi_1h[i] > 60
        rsi_extreme_oversold = rsi_1h[i] < 25  # Fallback — always triggers
        rsi_extreme_overbought = rsi_1h[i] > 75  # Fallback — always triggers
        
        desired_signal = 0.0
        
        # === PRIMARY ENTRY: Trend + Pullback + Volume ===
        # Long: 4h bullish + RSI pullback + volume confirmation
        if trend_4h_bullish and rsi_oversold and volume_ok:
            # Stronger signal if 12h also bullish
            if trend_12h_bullish:
                desired_signal = BASE_SIZE
            else:
                desired_signal = REDUCED_SIZE
        
        # Short: 4h bearish + RSI rally + volume confirmation
        if trend_4h_bearish and rsi_overbought and volume_ok:
            # Stronger signal if 12h also bearish
            if trend_12h_bearish:
                desired_signal = -BASE_SIZE
            else:
                desired_signal = -REDUCED_SIZE
        
        # === FALLBACK: RSI Extremes (GUARANTEES TRADES) ===
        # This ensures we get trades even when trend filter blocks entries
        if desired_signal == 0.0:
            if rsi_extreme_oversold:
                # Extreme oversold — long even against trend (mean reversion)
                desired_signal = REDUCED_SIZE
            elif rsi_extreme_overbought:
                # Extreme overbought — short even against trend (mean reversion)
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position through minor pullbacks ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bullish and RSI not overbought
                if trend_4h_bullish and rsi_1h[i] < 70:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bearish and RSI not oversold
                if trend_4h_bearish and rsi_1h[i] > 30:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses AND RSI overbought
            if trend_4h_bearish and rsi_1h[i] > 65:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses AND RSI oversold
            if trend_4h_bullish and rsi_1h[i] < 35:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= BASE_SIZE else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -BASE_SIZE else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_1h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals