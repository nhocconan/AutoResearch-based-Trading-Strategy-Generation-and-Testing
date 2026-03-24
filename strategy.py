#!/usr/bin/env python3
"""
Experiment #048: 30m Primary + 4h/1d HTF — Simplified RSI Pullback with Regime Filter

Hypothesis: Previous 30m strategies failed due to TOO STRICT entry conditions (0 trades).
This simplifies to proven pattern: HTF trend + RSI pullback + optional regime filter.

Key changes from failed #038/#045:
1. RSI(14) instead of CRSI - simpler, more reliable signals
2. LOOSE thresholds: RSI<35 long, RSI>65 short (vs 15/85)
3. Only 2 confluence required (not 3+): HTF trend + RSI extreme
4. Session/volume = soft bonus, NOT hard requirement
5. Funding fallback to 0 if data unavailable (prevents 0 trades)

Entry Logic:
- LONG: 4h HMA bullish + RSI(14)<35 + (optional: CHOP>50 or session 8-20 UTC)
- SHORT: 4h HMA bearish + RSI(14)>65 + (optional: CHOP>50 or session 8-20 UTC)
- Size: 0.25 base, 0.30 with 1d confirmation, 0.20 counter-HTF

Risk: 2.5x ATR trailing stop, discrete signal levels (0.0, ±0.20, ±0.25, ±0.30)
Target: 40-80 trades/year, Sharpe>0.4, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_rsi_pullback_4h1d_hma_loose_v1"
timeframe = "30m"
leverage = 1.0

def calculate_rsi(close, period=14):
    """RSI with proper min_periods"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_hma(close, period=21):
    """Hull Moving Average for trend"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index for regime detection"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high == lowest_low:
            chop[i] = 100.0
            continue
        
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            prev_close = close[j-1] if j > 0 else close[j]
            tr = max(high[j] - low[j], abs(high[j] - prev_close), abs(low[j] - prev_close))
            atr_sum += tr
        
        chop[i] = 100.0 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(close, period=200):
    """Simple Moving Average for major trend filter"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma

def load_funding_data(symbol):
    """Load funding rate data - returns None if unavailable (fallback to 0)"""
    try:
        import os
        symbol_base = symbol.replace('USDT', '').lower()
        funding_path = f"data/processed/funding/{symbol_base}.parquet"
        
        if os.path.exists(funding_path):
            df_funding = pd.read_parquet(funding_path)
            return {
                'timestamp': df_funding['timestamp'].values,
                'funding_rate': df_funding['funding_rate'].values
            }
    except Exception:
        pass
    
    return None

def get_funding_at_time(funding_data, timestamp):
    """Get funding rate closest to given timestamp"""
    if funding_data is None:
        return 0.0
    
    ts_arr = funding_data['timestamp']
    fr_arr = funding_data['funding_rate']
    
    idx = np.searchsorted(ts_arr, timestamp)
    if idx >= len(ts_arr):
        idx = len(ts_arr) - 1
    if idx < 0:
        idx = 0
    
    return fr_arr[idx]

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values if "volume" in prices.columns else np.ones(len(close))
    open_time = prices["open_time"].values if "open_time" in prices.columns else np.arange(len(close))
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for primary trend
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d HMA for major trend confirmation
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (30m) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    sma200 = calculate_sma(close, period=200)
    
    # Volume SMA for volume filter
    vol_sma = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Try to load funding data (fallback to None if unavailable)
    funding_data = None
    try:
        funding_data = load_funding_data("BTCUSDT")
    except Exception:
        funding_data = None
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    CONFIRMED_SIZE = 0.30
    REDUCED_SIZE = 0.20
    MAX_SIZE = 0.35
    
    # Position tracking for stoploss
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
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h and 1d HMA) ===
        hma_4h_bull = close[i] > hma_4h_aligned[i]
        hma_4h_bear = close[i] < hma_4h_aligned[i]
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # SMA200 for major trend filter
        sma200_bull = not np.isnan(sma200[i]) and close[i] > sma200[i]
        sma200_bear = not np.isnan(sma200[i]) and close[i] < sma200[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop[i] > 50.0
        is_trending = chop[i] < 45.0
        
        # === SESSION FILTER (8-20 UTC) - soft bonus, not hard requirement ===
        try:
            hour_utc = (open_time[i] // 3600000) % 24
            in_session = 8 <= hour_utc <= 20
        except Exception:
            in_session = True
        
        # === VOLUME FILTER - soft bonus ===
        vol_ratio = volume[i] / vol_sma[i] if vol_sma[i] > 1e-10 else 1.0
        vol_ok = vol_ratio > 0.8
        
        # === FUNDING RATE CONTRARIAN (optional bonus) ===
        funding_signal = 0.0
        try:
            funding_rate = get_funding_at_time(funding_data, open_time[i])
            if funding_rate > 0.01:
                funding_signal = -0.05
            elif funding_rate < -0.01:
                funding_signal = 0.05
        except Exception:
            funding_signal = 0.0
        
        # === ENTRY SIGNALS (LOOSE thresholds for trade generation) ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # Count HTF alignment
        htf_bull_count = sum([hma_4h_bull, hma_1d_bull, sma200_bull])
        htf_bear_count = sum([hma_4h_bear, hma_1d_bear, sma200_bear])
        
        # LONG ENTRY: 4h HMA bullish + RSI<35 (oversold pullback)
        if hma_4h_bull and rsi[i] < 35.0:
            # Base size with 4h trend
            signal_strength = BASE_SIZE
            
            # Increase size with 1d confirmation
            if hma_1d_bull:
                signal_strength = CONFIRMED_SIZE
            
            # Bonus for session/volume (soft, not required)
            if in_session and vol_ok:
                signal_strength = min(signal_strength + 0.02, MAX_SIZE)
            
            desired_signal = signal_strength + funding_signal
        
        # SHORT ENTRY: 4h HMA bearish + RSI>65 (overbought pullback)
        elif hma_4h_bear and rsi[i] > 65.0:
            # Base size with 4h trend
            signal_strength = BASE_SIZE
            
            # Increase size with 1d confirmation
            if hma_1d_bear:
                signal_strength = CONFIRMED_SIZE
            
            # Bonus for session/volume (soft, not required)
            if in_session and vol_ok:
                signal_strength = min(signal_strength + 0.02, MAX_SIZE)
            
            desired_signal = -signal_strength + funding_signal
        
        # === CHOPPY REGIME: Allow counter-HTF trades with reduced size ===
        if is_choppy and desired_signal == 0.0:
            # Mean reversion in range: RSI extremes without HTF requirement
            if rsi[i] < 30.0:
                desired_signal = REDUCED_SIZE + funding_signal
            elif rsi[i] > 70.0:
                desired_signal = -REDUCED_SIZE + funding_signal
        
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
        
        # === DISCRETIZE SIGNAL VALUES ===
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= CONFIRMED_SIZE * 0.85:
            final_signal = CONFIRMED_SIZE
        elif desired_signal <= -CONFIRMED_SIZE * 0.85:
            final_signal = -CONFIRMED_SIZE
        elif desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
        elif abs(desired_signal) >= 0.08:
            final_signal = np.sign(desired_signal) * REDUCED_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
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
        
        signals[i] = final_signal
    
    return signals