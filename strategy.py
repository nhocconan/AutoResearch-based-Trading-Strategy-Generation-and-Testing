#!/usr/bin/env python3
"""
Experiment #317: 15m Primary + 4h/12h HTF — RSI Pullback with HTF Trend Filter v1

Hypothesis: 15m strategies fail due to OVER-FILTERING (0 trades). This uses SIMPLER logic:
- 4h HMA for trend direction (proven edge from best strategies)
- 15m RSI for entry timing (oversold bounce in uptrend, overbought fade in downtrend)
- Volume confirmation (above average = institutional participation)
- Session filter: 00-12 UTC (London/NY overlap = highest crypto volume)

Key differences from failed 15m attempts:
1. LOOSENED RSI: <35/>65 instead of <20/>80 (ensures trades generate)
2. SINGLE HTF FILTER: 4h HMA only (not 4h+12h+1d which caused 0 trades)
3. VOLUME THRESHOLD: 1.2x average (not 2.0x which was too restrictive)
4. DISCRETE SIZING: 0.20 base, 0.30 when HTF strongly aligned

Target: 50-100 trades/year, Sharpe>0.40, DD>-30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_rsi_pullback_4h_trend_volume_session_v1"
timeframe = "15m"
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

def calculate_volume_ma(volume, period=20):
    """Volume Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

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
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (15m) indicators
    rsi = calculate_rsi(close, period=14)
    rsi_fast = calculate_rsi(close, period=7)
    atr = calculate_atr(high, low, close, period=14)
    vol_ma = calculate_volume_ma(volume, period=20)
    
    # Price relative to 4h HMA (percentage)
    price_vs_4h_hma = np.zeros(n)
    price_vs_4h_hma[:] = np.nan
    for i in range(n):
        if not np.isnan(hma_4h_aligned[i]) and hma_4h_aligned[i] > 1e-10:
            price_vs_4h_hma[i] = (close[i] - hma_4h_aligned[i]) / hma_4h_aligned[i] * 100.0
    
    signals = np.zeros(n)
    SIZE_BASE = 0.20
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (00-12 UTC preferred) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        is_prime_session = 0 <= hour_utc <= 12
        
        # === HTF TREND BIAS ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # 12h for stronger confirmation
        htf_12h_bull = not np.isnan(hma_12h_aligned[i]) and close[i] > hma_12h_aligned[i]
        htf_12h_bear = not np.isnan(hma_12h_aligned[i]) and close[i] < hma_12h_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = False
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 1e-10:
            volume_confirmed = volume[i] > 1.2 * vol_ma[i]
        
        # === RSI EXTREMES (LOOSENED for trade generation) ===
        rsi_oversold = rsi[i] < 35.0  # Was 30, loosened
        rsi_overbought = rsi[i] > 65.0  # Was 70, loosened
        
        # Fast RSI for momentum confirmation
        rsi_fast_oversold = not np.isnan(rsi_fast[i]) and rsi_fast[i] < 30.0
        rsi_fast_overbought = not np.isnan(rsi_fast[i]) and rsi_fast[i] > 70.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: RSI oversold + price above 4h HMA (pullback in uptrend)
        if rsi_oversold and htf_4h_bull:
            # Base entry: just RSI + HTF trend
            desired_signal = SIZE_BASE
            
            # Boost if: volume confirmed OR prime session OR 12h aligned
            if volume_confirmed or is_prime_session or htf_12h_bull:
                desired_signal = SIZE_STRONG
            
            # Extra boost if fast RSI also oversold (momentum confirmation)
            if rsi_fast_oversold:
                desired_signal = SIZE_STRONG
        
        # SHORT: RSI overbought + price below 4h HMA (rally in downtrend)
        elif rsi_overbought and htf_4h_bear:
            # Base entry: just RSI + HTF trend
            desired_signal = -SIZE_BASE
            
            # Boost if: volume confirmed OR prime session OR 12h aligned
            if volume_confirmed or is_prime_session or htf_12h_bear:
                desired_signal = -SIZE_STRONG
            
            # Extra boost if fast RSI also overbought (momentum confirmation)
            if rsi_fast_overbought:
                desired_signal = -SIZE_STRONG
        
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