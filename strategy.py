#!/usr/bin/env python3
"""
Experiment #1135: 1h Primary + 4h/1d HTF — Simplified HMA Trend + RSI Pullback + Session

Hypothesis: After 826+ failed experiments, the critical insight is BALANCE between
trade frequency and filter strictness. For 1h timeframe:
- Use 4h HMA(21) for trend direction (faster than 1d, better for 1h entries)
- Use 1h RSI(14) for pullback timing with LOOSE thresholds (40/60 not 30/70)
- Add session filter (8-20 UTC) to avoid low-volume Asian session whipsaws
- Add volume filter (>0.7x avg) but NOT too strict
- ATR(14) 2.0x trailing stop (tighter than 4h strategy due to lower TF)
- Position size 0.25 (smaller than 4h due to more trades)

Why this should beat Sharpe=0.612:
- 4h HMA is more responsive than 1d for 1h entries (better timing)
- Session filter removes 40% of low-quality trades (Asian session chop)
- Volume filter ensures liquidity without being too restrictive
- Simpler than CRSI/Chop regimes that caused 0 trades in #1128, #1130, #1132
- Hold logic maintains positions through trend continuations

Timeframe: 1h (primary)
HTF: 4h — loaded ONCE before loop using mtf_data helper
Position Size: 0.25 base (discrete: 0.0, ±0.25)
Stoploss: 2.0x ATR trailing
Target: 40-80 trades/year, Sharpe > 0.612
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h_session_vol_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average — reduces lag while maintaining smoothness.
    Formula: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    def wma(data, span):
        """Weighted Moving Average."""
        if span < 1:
            span = 1
        result = np.full(len(data), np.nan)
        weights = np.arange(1, span + 1, dtype=np.float64)
        for i in range(span - 1, len(data)):
            window = data[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    half = max(1, int(period / 2))
    sqrt_period = max(1, int(np.sqrt(period)))
    
    wma1 = wma(close, half)
    wma2 = wma(close, period)
    
    diff = 2 * wma1 - wma2
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    diff = np.diff(close)
    gain = np.where(diff > 0, diff, 0.0)
    loss = np.where(diff < 0, -diff, 0.0)
    
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 1e-10
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    rsi[~mask] = 50.0
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_sma(data, period=20):
    """Simple Moving Average."""
    n = len(data)
    sma = np.full(n, np.nan)
    if n < period:
        return sma
    for i in range(period - 1, n):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1h) indicators
    rsi_1h = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(rsi_1h[i]) or np.isnan(atr[i]) or np.isnan(vol_sma[i]):
            continue
        if np.isnan(hma_4h_aligned[i]):
            continue
        if atr[i] <= 1e-10 or vol_sma[i] <= 1e-10:
            continue
        
        # === EXTRACT HOUR FROM TIMESTAMP (UTC) ===
        # open_time is in milliseconds
        hour_utc = (open_time[i] // (1000 * 60 * 60)) % 24
        
        # === SESSION FILTER (8-20 UTC) ===
        # Avoid Asian session chop (0-8 UTC) and late night (20-24 UTC)
        in_session = 8 <= hour_utc <= 20
        
        # === VOLUME FILTER ===
        # Volume > 0.7x 20-period average (not too strict)
        volume_ok = volume[i] > 0.7 * vol_sma[i]
        
        # === TREND DIRECTION (4h HMA) ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === PULLBACK SIGNAL (1h RSI) ===
        # LOOSE thresholds: 40/60 (not 30/70) to ensure trade frequency
        rsi_oversold = rsi_1h[i] < 45.0
        rsi_overbought = rsi_1h[i] > 55.0
        
        desired_signal = 0.0
        
        # === LONG ENTRY ===
        # 4h trend bull + 1h RSI pullback + session + volume
        if trend_bull and rsi_oversold and in_session and volume_ok:
            desired_signal = BASE_SIZE
        
        # === SHORT ENTRY ===
        # 4h trend bear + 1h RSI pullback + session + volume
        elif trend_bear and rsi_overbought and in_session and volume_ok:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.0x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.0 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.0 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if trend intact ===
        # Critical: don't exit just because RSI moved, hold through trend
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h trend still bull
                if trend_bull:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if 4h trend still bear
                if trend_bear:
                    desired_signal = -BASE_SIZE
        
        # === EXIT CONDITIONS ===
        # Exit when 4h trend reverses
        if in_position and position_side > 0:
            # Exit long if 4h trend reverses to bear
            if trend_bear:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if 4h trend reverses to bull
            if trend_bull:
                desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Flip position
                position_side = int(np.sign(desired_signal))
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
        
        signals[i] = desired_signal
    
    return signals