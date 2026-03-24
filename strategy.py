#!/usr/bin/env python3
"""
Experiment #1500: 1h Primary + 4h/12h HTF — Regime-Adaptive HMA Trend with RSI Pullback

Hypothesis: After 1100+ failed strategies, the pattern for 1h timeframe is clear:
1. Previous 1h strategies (exp #1490, #1495, #1498) got Sharpe=0.000 because entry conditions TOO STRICT
2. Choppiness Index + Connors RSI combination generates ZERO trades on 1h
3. Solution: LOOSE entry conditions with HTF trend bias + simple RSI pullback
4. 12h HMA for macro direction, 4h HMA for confirmation, 1h RSI for entry timing
5. Target: 40-80 trades/year (NOT 0!), use discrete signal sizes (0.0, ±0.25, ±0.35)

Key design choices:
- LOOSE RSI bands: Long when RSI(14) < 55 in uptrend, Short when RSI(14) > 45 in downtrend
- Only 2 HTF filters: 12h HMA(21) + 4h HMA(21) — more filters = 0 trades
- Volume filter: only 1.0x average (not 1.5x which kills trades)
- Session filter: 8-20 UTC (liquid hours) — but NOT too restrictive
- ATR(14) 2.5x trailing stop for risk management
- Position size 0.30 with discrete levels to minimize fee churn

Timeframe: 1h (as required by experiment)
HTF: 4h and 12h (call get_htf_data ONCE before loop for each!)
Position Size: 0.30 max (discrete: 0.0, ±0.20, ±0.30)
Target: 50-100 trades/train, 10-20 trades/test, Sharpe > 0.618
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_4h12h_regime_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            if np.any(np.isnan(data[i - w_period + 1:i + 1])):
                continue
            result[i] = np.sum(data[i - w_period + 1:i + 1] * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
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

def calculate_sma(close, period=50):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_sma
    ratio[:period] = np.nan
    return ratio

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
    
    # Calculate and align HTF HMAs for trend bias
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (1h) indicators
    hma_1h = calculate_hma(close, period=21)
    sma_50 = calculate_sma(close, period=50)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(rsi[i]) or np.isnan(hma_1h[i]):
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
        if np.isnan(sma_50[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC) — liquid hours only ===
        hour = (open_time[i] // 3600000) % 24
        in_session = 8 <= hour <= 20
        
        # === MACRO TREND (12h HMA) — direction bias ONLY ===
        h12_bull = close[i] > hma_12h_aligned[i]
        h12_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) — confirmation ===
        h4_bull = close[i] > hma_4h_aligned[i]
        h4_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (1h HMA) — entry timing ===
        h1_bull = close[i] > hma_1h[i]
        h1_bear = close[i] < hma_1h[i]
        
        # === SMA50 FILTER ===
        above_sma50 = close[i] > sma_50[i]
        below_sma50 = close[i] < sma_50[i]
        
        # === VOLUME FILTER (loose: >0.8x avg) ===
        volume_ok = vol_ratio[i] > 0.8
        
        # === RSI PULLBACK — LOOSE bands for MORE trades ===
        # Long: RSI pulled back but still bullish (>40)
        # Short: RSI rallied but still bearish (<60)
        rsi_long_setup = rsi[i] > 40.0 and rsi[i] < 65.0
        rsi_short_setup = rsi[i] > 35.0 and rsi[i] < 60.0
        
        # === DESIRED SIGNAL — LOOSE conditions to ensure trades ===
        desired_signal = 0.0
        
        # LONG: 12h bull + 4h bull + 1h setup + volume + session
        # Option 1: All 3 TF aligned (strongest)
        if h12_bull and h4_bull and h1_bull and rsi_long_setup and volume_ok and in_session:
            desired_signal = BASE_SIZE
        # Option 2: 12h + 4h aligned (more frequent)
        elif h12_bull and h4_bull and rsi[i] > 45.0 and volume_ok:
            desired_signal = BASE_SIZE * 0.8
        # Option 3: Just 12h trend + RSI (loosest, ensures trades)
        elif h12_bull and rsi[i] > 42.0 and above_sma50:
            desired_signal = BASE_SIZE * 0.6
        
        # SHORT: 12h bear + 4h bear + 1h setup + volume + session
        # Option 1: All 3 TF aligned (strongest)
        elif h12_bear and h4_bear and h1_bear and rsi_short_setup and volume_ok and in_session:
            desired_signal = -BASE_SIZE
        # Option 2: 12h + 4h aligned (more frequent)
        elif h12_bear and h4_bear and rsi[i] < 55.0 and volume_ok:
            desired_signal = -BASE_SIZE * 0.8
        # Option 3: Just 12h trend + RSI (loosest, ensures trades)
        elif h12_bear and rsi[i] < 58.0 and below_sma50:
            desired_signal = -BASE_SIZE * 0.6
        
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
        if desired_signal >= BASE_SIZE * 0.7:
            final_signal = BASE_SIZE
        elif desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE * 0.8
        elif desired_signal >= BASE_SIZE * 0.3:
            final_signal = BASE_SIZE * 0.6
        elif desired_signal <= -BASE_SIZE * 0.7:
            final_signal = -BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.5:
            final_signal = -BASE_SIZE * 0.8
        elif desired_signal <= -BASE_SIZE * 0.3:
            final_signal = -BASE_SIZE * 0.6
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