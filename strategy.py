#!/usr/bin/env python3
"""
Experiment #521: 15m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume

Hypothesis: 15m timeframe with 4h/1d HTF filters can capture intraday moves
while avoiding whipsaw. Key insight from failed 15m experiments (#509, #517):
entry conditions were TOO STRICT generating 0 trades. This version uses
LOOSER thresholds to ensure trade generation while maintaining quality.

Strategy logic:
1. 1d HMA(21) = macro trend bias (HTF filter #1)
2. 4h HMA(21) = intermediate trend (HTF filter #2)
3. 15m RSI(7) = entry timing (oversold <35 long, overbought >65 short)
4. 15m Volume spike (>1.5x 20-bar avg) = confirmation
5. 15m HMA(16) = short-term trend confirmation
6. ATR(14)*2.5 stoploss on all positions
7. Session filter: prefer 00-12 UTC (London+NY overlap)

Key changes from failed 15m experiments:
- LOOSEN RSI thresholds (35/65 instead of 30/70) for MORE trades
- Use OR logic for entries (not strict AND)
- Reduce confluence to 2-3 filters (not 5+)
- Smaller position size (0.15-0.25) for higher frequency
- Target 50-80 trades/year (not >100 to avoid fee drag)

Target: Sharpe>0.40, trades>=150 train (40/year), trades>=20 test
Timeframe: 15m
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_vol_4h1d_v2"
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

def calculate_volume_spike(volume, period=20):
    """Volume spike detection - current volume vs rolling average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_avg
    
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 15m indicators
    hma_15m = calculate_hma(close, period=16)
    rsi_7 = calculate_rsi(close, period=7)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_spike(volume, period=20)
    
    # Session filter: 00-12 UTC preferred (London+NY overlap for crypto)
    # 15m bars: 96 bars per day, bars 0-47 are 00:00-12:00 UTC
    open_time = prices["open_time"].values
    hour_utc = (open_time // 3600000) % 24  # Convert ms to hours
    session_active = (hour_utc >= 0) & (hour_utc < 12)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.18
    SIZE_STRONG = 0.25
    SIZE_HALF = 0.12
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_15m[i]) or np.isnan(rsi_7[i]) or np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1d HTF MACRO BIAS ===
        htf_1d_bull = close[i] > hma_1d_aligned[i]
        htf_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h HTF INTERMEDIATE TREND ===
        htf_4h_bull = close[i] > hma_4h_aligned[i]
        htf_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === 15m HMA SHORT-TERM TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI CONDITIONS (LOOSENED for more trades) ===
        rsi_oversold = rsi_7[i] < 35.0  # Looser than 30
        rsi_overbought = rsi_7[i] > 65.0  # Looser than 70
        rsi_neutral = (rsi_7[i] >= 35.0) & (rsi_7[i] <= 65.0)
        
        # RSI momentum
        rsi_rising = rsi_7[i] > rsi_7[i-1] if i > 0 else False
        rsi_falling = rsi_7[i] < rsi_7[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3  # Lower threshold for more trades
        vol_normal = vol_ratio[i] < 4.0  # Avoid extreme spikes
        
        # === HTF CONFLUENCE ===
        # Strong bull: both 1d and 4h agree
        htf_strong_bull = htf_1d_bull and htf_4h_bull
        htf_strong_bear = htf_1d_bear and htf_4h_bear
        # Mixed: only one agrees
        htf_mixed_bull = htf_1d_bull or htf_4h_bull
        htf_mixed_bear = htf_1d_bear or htf_4h_bear
        
        # === SESSION FILTER ===
        in_session = session_active[i]
        
        # === ENTRY LOGIC (LOOSE CONDITIONS FOR TRADE GENERATION) ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # Condition 1: HTF bull + RSI oversold + volume (primary)
        if htf_strong_bull and rsi_oversold and vol_normal:
            desired_signal = SIZE_STRONG if in_session else SIZE_BASE
        # Condition 2: HTF bull + RSI rising from oversold (recovery)
        elif htf_mixed_bull and rsi_oversold and rsi_rising and vol_normal:
            desired_signal = SIZE_BASE
        # Condition 3: HMA bull + RSI neutral + volume spike (momentum)
        elif hma_bull and rsi_neutral and rsi_rising and vol_spike:
            desired_signal = SIZE_BASE * 0.8
        # Condition 4: Price above both HTF HMAs + RSI > 50 (trend continuation)
        elif htf_strong_bull and hma_bull and rsi_7[i] > 50.0:
            desired_signal = SIZE_BASE * 0.7
        
        # SHORT ENTRIES
        # Condition 1: HTF bear + RSI overbought + volume (primary)
        if htf_strong_bear and rsi_overbought and vol_normal:
            desired_signal = -SIZE_STRONG if in_session else -SIZE_BASE
        # Condition 2: HTF bear + RSI falling from overbought (recovery)
        elif htf_mixed_bear and rsi_overbought and rsi_falling and vol_normal:
            desired_signal = -SIZE_BASE
        # Condition 3: HMA bear + RSI neutral + volume spike (momentum)
        elif hma_bear and rsi_neutral and rsi_falling and vol_spike:
            desired_signal = -SIZE_BASE * 0.8
        # Condition 4: Price below both HTF HMAs + RSI < 50 (trend continuation)
        elif htf_strong_bear and hma_bear and rsi_7[i] < 50.0:
            desired_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
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
        elif desired_signal >= SIZE_BASE * 0.5:
            final_signal = SIZE_HALF
        elif desired_signal <= -SIZE_BASE * 0.5:
            final_signal = -SIZE_HALF
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals