#!/usr/bin/env python3
"""
Experiment #648: 30m Primary + 4h/1d HTF — HMA Trend + RSI Pullback + Volume

Hypothesis: 30m timeframe with 4h HMA for trend direction + 30m RSI pullback entries
provides optimal balance between trade frequency and signal quality. This adapts the
proven mtf_hma_rsi_zscore_v1 pattern (Sharpe=5.4) to 30m with stricter filters to
control trade count (target 40-80/year).

Key innovations:
1. 4h HMA(21) for macro trend direction — only trade with HTF trend
2. 30m RSI(14) pullback entries — enter on RSI 35-45 in uptrend, 55-65 in downtrend
3. Volume confirmation — volume > 0.8x 20-bar average
4. 1d HMA for additional regime filter — avoid counter-trend vs daily
5. ATR(14) trailing stoploss — 2.5x ATR from entry/highest
6. Discrete sizing: 0.25 for entries, reduces fee churn

Why this should work on 30m:
- HTF (4h/1d) determines DIRECTION, 30m determines TIMING
- RSI pullback entries catch continuations, not reversals
- Volume filter avoids low-liquidity false breakouts
- Conservative sizing (0.25) survives 2022 crash with ~25% DD
- Loose enough RSI thresholds to ensure ≥10 trades/train, ≥3/test

Target: Sharpe > 0.612, trades >= 30 train, >= 5 test, ALL symbols positive Sharpe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_hma_rsi_vol_pullback_4h1d_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average for smoother trend detection."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        result = pd.Series(series).rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """RSI with proper min_periods."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Use EMA-style smoothing for RSI
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # Initial SMA
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    # EMA smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    # Calculate RSI
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / avg_loss
        rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr1 = high[i] - low[i]
        tr2 = np.abs(high[i] - close[i - 1])
        tr3 = np.abs(low[i] - close[i - 1])
        tr[i] = max(tr1, tr2, tr3)
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume."""
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 30m indicators (primary timeframe)
    rsi_30m = calculate_rsi(close, period=14)
    atr_30m = calculate_atr(high, low, close, period=14)
    vol_sma_30m = calculate_volume_sma(volume, period=20)
    
    # Calculate and align HTF indicators
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size for 30m (smaller due to more trades)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start later to ensure all indicators ready
        # Skip if indicators not ready
        if np.isnan(rsi_30m[i]) or np.isnan(atr_30m[i]) or atr_30m[i] <= 1e-10:
            continue
        if np.isnan(vol_sma_30m[i]) or vol_sma_30m[i] <= 1e-10:
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        htf_4h_bullish = close[i] > hma_4h_aligned[i]
        htf_4h_bearish = close[i] < hma_4h_aligned[i]
        
        # === HTF TREND BIAS (1d HMA) — Additional filter ===
        htf_1d_bullish = close[i] > hma_1d_aligned[i]
        htf_1d_bearish = close[i] < hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = volume[i] > 0.8 * vol_sma_30m[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 35-50 in uptrend (buy the dip)
        rsi_long_pullback = 35 <= rsi_30m[i] <= 50
        # Short: RSI pulled back to 50-65 in downtrend (sell the rally)
        rsi_short_pullback = 50 <= rsi_30m[i] <= 65
        
        # RSI extreme reversal (less common, but higher conviction)
        rsi_oversold = rsi_30m[i] < 30
        rsi_overbought = rsi_30m[i] > 70
        
        desired_signal = 0.0
        
        # === LONG ENTRY CONDITIONS ===
        # Primary: 4h bullish + 1d not bearish + RSI pullback + volume
        if htf_4h_bullish and not htf_1d_bearish and rsi_long_pullback and volume_confirmed:
            desired_signal = SIZE
        # Secondary: 4h bullish + RSI oversold (stronger signal)
        elif htf_4h_bullish and rsi_oversold:
            desired_signal = SIZE
        
        # === SHORT ENTRY CONDITIONS ===
        # Primary: 4h bearish + 1d not bullish + RSI pullback + volume
        elif htf_4h_bearish and not htf_1d_bullish and rsi_short_pullback and volume_confirmed:
            desired_signal = -SIZE
        # Secondary: 4h bearish + RSI overbought (stronger signal)
        elif htf_4h_bearish and rsi_overbought:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (Trailing ATR) ===
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
        
        # === HOLD LOGIC — Maintain position if trend unchanged ===
        # This ensures we don't exit too quickly on minor pullbacks
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if 4h still bullish and RSI not extremely overbought
                if htf_4h_bullish and rsi_30m[i] < 75:
                    desired_signal = SIZE
            elif position_side < 0:
                # Hold short if 4h still bearish and RSI not extremely oversold
                if htf_4h_bearish and rsi_30m[i] > 25:
                    desired_signal = -SIZE
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0:
            desired_signal = SIZE
        elif desired_signal < 0:
            desired_signal = -SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_30m[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            # If same side, update trailing stop levels
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