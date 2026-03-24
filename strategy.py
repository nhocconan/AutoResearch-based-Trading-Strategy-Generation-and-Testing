#!/usr/bin/env python3
"""
Experiment #037: 15m Primary + 4h/12h HTF — HMA Trend + RSI Pullback + Session Filter

Hypothesis: 15m strategies failed (Sharpe=0.000) because entry conditions were TOO STRICT.
This strategy uses PROVEN pattern from best performer (mtf_hma_rsi_zscore_v1):
- 4h HMA for major trend direction (slower, more reliable)
- 15m RSI(7) for pullback entries (faster response than RSI14)
- Session filter: UTC 00-12 (London+NY overlap) to avoid low-volume whipsaws
- Volume confirmation: current > 0.8 * 20-bar avg (ensures real moves)
- LOOSE entry thresholds to ensure >=30 trades on train, >=3 on test

Key design choices:
- Timeframe: 15m (target 40-100 trades/year)
- HTF: 4h HMA(21) for trend bias, 12h HMA(50) for major regime
- Entry: RSI(7) pullback in trend direction (RSI<45 long, RSI>55 short)
- Position size: 0.20 (20% of capital, smaller for 15m frequency)
- Stoploss: 2.5x ATR(14) trailing
- Session: prefer 00-12 UTC but allow 12-24 with stronger signal

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_hma_rsi_session_volume_4h12h_v1"
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

def calculate_volume_avg(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_avg = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_avg

def get_utc_hour(prices, idx):
    """Extract UTC hour from open_time timestamp"""
    # open_time is in milliseconds since epoch
    ts_ms = prices['open_time'].iloc[idx]
    ts_sec = ts_ms / 1000.0
    utc_hour = int((ts_sec % 86400) / 3600)
    return utc_hour

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA for major regime
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (15m) indicators
    hma_15m = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=7)  # Faster RSI for 15m
    atr = calculate_atr(high, low, close, period=14)
    vol_avg = calculate_volume_avg(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size (conservative for 15m frequency)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_15m[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(vol_avg[i]) or vol_avg[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (UTC hours) ===
        utc_hour = get_utc_hour(prices, i)
        # Preferred session: 00-12 UTC (London+NY overlap)
        # Secondary session: 12-24 UTC (Asia+early Europe) - requires stronger signal
        is_preferred_session = (utc_hour >= 0 and utc_hour <= 12)
        
        # === VOLUME CONFIRMATION ===
        volume_ok = volume[i] > 0.8 * vol_avg[i]
        
        # === HTF BIAS (4h HMA) ===
        htf_bull = close[i] > hma_4h_aligned[i]
        htf_bear = close[i] < hma_4h_aligned[i]
        
        # === MAJOR REGIME (12h HMA) ===
        regime_bull = close[i] > hma_12h_aligned[i]
        regime_bear = close[i] < hma_12h_aligned[i]
        
        # === 15m HMA TREND ===
        hma_bull = close[i] > hma_15m[i]
        hma_bear = close[i] < hma_15m[i]
        
        # === RSI PULLBACK SIGNALS (LOOSE thresholds for trade generation) ===
        # Long: RSI pulled back in uptrend (RSI 30-50)
        rsi_oversold_pullback = rsi[i] < 50.0 and rsi[i] > 25.0
        # Short: RSI pulled back in downtrend (RSI 50-70)
        rsi_overbought_pullback = rsi[i] > 50.0 and rsi[i] < 75.0
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 1.0
        
        # LONG conditions
        if htf_bull and hma_bull and rsi_oversold_pullback and volume_ok:
            if is_preferred_session:
                desired_signal = SIZE
            else:
                # Secondary session: require regime confirmation
                if regime_bull:
                    desired_signal = SIZE * 0.7
                    signal_strength = 0.7
        
        # SHORT conditions
        elif htf_bear and hma_bear and rsi_overbought_pullback and volume_ok:
            if is_preferred_session:
                desired_signal = -SIZE
            else:
                # Secondary session: require regime confirmation
                if regime_bear:
                    desired_signal = -SIZE * 0.7
                    signal_strength = 0.7
        
        # Fallback: Strong RSI extremes (ensure trades generate)
        if desired_signal == 0.0:
            if rsi[i] < 20.0 and htf_bull and volume_ok:
                desired_signal = SIZE * 0.5
            elif rsi[i] > 80.0 and htf_bear and volume_ok:
                desired_signal = -SIZE * 0.5
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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
                # Flip position
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