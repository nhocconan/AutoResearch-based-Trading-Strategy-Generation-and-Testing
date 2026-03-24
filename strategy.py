#!/usr/bin/env python3
"""
Experiment #424: 12h Primary + 1d/1w HTF — Simplified HMA Trend Pullback v3

Hypothesis: Previous strategies failed due to overly complex regime detection
(ADX+CHOP combinations) that rarely triggered entries. This version SIMPLIFIES
to pure HMA trend alignment across 3 timeframes (1w bias → 1d confirm → 12h entry).

Key changes from #352:
1. REMOVED ADX/CHOP regime detection (too restrictive, caused 0 trades)
2. Pure HMA trend alignment: 1w > 1d > 12h for long, reverse for short
3. RSI pullback entry (30/70 thresholds) — looser than before
4. Fewer confluence requirements (max 3 filters per entry)
5. Better position tracking with proper stoploss reset on exit

Entry Logic (SIMPLIFIED):
- Long: 1w HMA bull + 1d HMA bull + 12h HMA bull + RSI < 50 (pullback)
- Short: 1w HMA bear + 1d HMA bear + 12h HMA bear + RSI > 50 (pullback)
- Add volume confirmation for stronger signals

Position sizing: 0.25 base, 0.30 when all 3 HTF aligned
Stoploss: 2.5x ATR(14) from entry price
Target: Sharpe>0.45, DD>-35%, trades>=30 train, trades>=5 test, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_trend_pullback_1d1w_v3"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_sma(volume, period=20):
    """Volume SMA for confirmation"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
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
    
    # Calculate and align HTF HMA for trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    hma_12h = calculate_hma(close, period=21)
    hma_12h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    vol_sma = calculate_volume_sma(volume, 20)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_12h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1w - slowest, most reliable) ===
        htf_1w_bull = close[i] > hma_1w_aligned[i]
        htf_1w_bear = close[i] < hma_1w_aligned[i]
        
        # === HTF TREND CONFIRM (1d - intermediate) ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 12h HMA TREND ===
        hma_bull = close[i] > hma_12h[i]
        hma_bear = close[i] < hma_12h[i]
        
        # === HMA CROSSOVER (faster signal) ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_12h_fast[i]) and not np.isnan(hma_12h_fast[i-1]):
            if not np.isnan(hma_12h[i]) and not np.isnan(hma_12h[i-1]):
                if hma_12h_fast[i-1] <= hma_12h[i-1] and hma_12h_fast[i] > hma_12h[i]:
                    hma_cross_long = True
                if hma_12h_fast[i-1] >= hma_12h[i-1] and hma_12h_fast[i] < hma_12h[i]:
                    hma_cross_short = True
        
        # === SMA200 FILTER (long-term trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_confirm = volume[i] > 1.2 * vol_sma[i]
        
        # === RSI PULLBACK (LOOSENED for more trades) ===
        rsi_pullback_long = rsi[i] < 50.0  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 50.0  # Pullback in downtrend
        
        # === ENTRY LOGIC (SIMPLIFIED - pure trend alignment) ===
        desired_signal = 0.0
        
        # LONG: All HTF aligned bull + RSI pullback
        # Condition 1: 1w bull (primary bias)
        # Condition 2: 1d bull (confirmation)
        # Condition 3: 12h bull OR HMA cross long (entry trigger)
        # Condition 4: RSI < 50 (pullback entry, not chasing)
        if htf_1w_bull and htf_1d_bull:
            if (hma_bull or hma_cross_long) and rsi_pullback_long:
                desired_signal = SIZE_STRONG if vol_confirm else SIZE_BASE
        
        # SHORT: All HTF aligned bear + RSI pullback
        # Condition 1: 1w bear (primary bias)
        # Condition 2: 1d bear (confirmation)
        # Condition 3: 12h bear OR HMA cross short (entry trigger)
        # Condition 4: RSI > 50 (pullback entry, not chasing)
        elif htf_1w_bear and htf_1d_bear:
            if (hma_bear or hma_cross_short) and rsi_pullback_short:
                desired_signal = -SIZE_STRONG if vol_confirm else -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals