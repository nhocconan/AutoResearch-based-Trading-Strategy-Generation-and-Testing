#!/usr/bin/env python3
"""
Experiment #256: 30m Primary + 4h/1d HTF — RSI Pullback + Volume Spike + HTF Trend

Hypothesis: 30m timeframe can work if we use HTF (4h/1d) for TREND DIRECTION
and 30m only for ENTRY TIMING. Key insight from failures: too many filters = 0 trades.

This strategy combines:
1. 4h HMA(21) for primary trend bias (long only when 4h HMA bullish)
2. 1d HMA(50) for major trend confirmation
3. 30m RSI(14) pullback entries (RSI<40 long, RSI>60 short) - LOOSENED thresholds
4. Volume spike confirmation (vol > 1.3x 20-bar avg) - optional, not required
5. ATR(14) trailing stoploss at 2.5x

CRITICAL CHANGES FROM FAILED ATTEMPTS:
- RSI thresholds LOOSENED: 40/60 instead of 30/70 (ensures trades)
- Volume filter is OPTIONAL confluence, not required (prevents 0 trades)
- No session filter (previous attempts with session = 0 trades)
- Position size: 0.25 base, 0.30 strong (discrete levels)
- Stoploss: 2.5x ATR trailing

Target: 40-80 trades/year, Sharpe>0.40, DD>-40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_hma_4h1d_vol_v1"
timeframe = "30m"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    # Volume SMA for spike detection
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
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
        
        # === HTF TREND BIAS ===
        # 4h HMA for intermediate trend
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # 1d HMA for major trend (optional confluence)
        htf_1d_valid = not np.isnan(hma_1d_aligned[i])
        htf_1d_bull = htf_1d_valid and close[i] > hma_1d_aligned[i]
        htf_1d_bear = htf_1d_valid and close[i] < hma_1d_aligned[i]
        
        # === SMA200 FILTER ===
        above_sma200 = not np.isnan(sma_200[i]) and close[i] > sma_200[i]
        below_sma200 = not np.isnan(sma_200[i]) and close[i] < sma_200[i]
        
        # === RSI PULLBACK (LOOSENED THRESHOLDS) ===
        rsi_oversold = rsi[i] < 40.0  # was 30, now 40 for more trades
        rsi_overbought = rsi[i] > 60.0  # was 70, now 60 for more trades
        
        # === VOLUME SPIKE (OPTIONAL CONFLUENCE) ===
        vol_spike = False
        if not np.isnan(vol_sma[i]) and vol_sma[i] > 1e-10:
            vol_spike = volume[i] > 1.3 * vol_sma[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 4h bullish + RSI pullback + (optional: volume spike OR 1d bullish)
        if htf_4h_bull and rsi_oversold:
            # Require at least one additional confluence
            if vol_spike or (htf_1d_valid and htf_1d_bull) or above_sma200:
                # Strong signal if multiple confluence
                confluence_count = sum([vol_spike, htf_1d_bull, above_sma200])
                if confluence_count >= 2:
                    desired_signal = SIZE_STRONG
                else:
                    desired_signal = SIZE_BASE
        
        # SHORT: 4h bearish + RSI overbought + (optional: volume spike OR 1d bearish)
        elif htf_4h_bear and rsi_overbought:
            if vol_spike or (htf_1d_valid and htf_1d_bear) or below_sma200:
                confluence_count = sum([vol_spike, htf_1d_bear, below_sma200])
                if confluence_count >= 2:
                    desired_signal = -SIZE_STRONG
                else:
                    desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
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
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i] if position_side > 0 else 0.0
                lowest_since_entry = low[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals