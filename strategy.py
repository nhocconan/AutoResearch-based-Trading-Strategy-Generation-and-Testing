#!/usr/bin/env python3
"""
Experiment #1187: 1d Primary + 4h HTF — Vol Spike Reversion + HMA Trend Filter

Hypothesis: Recent 1d strategies (#1183 Sharpe=0.334) show promise but need better entry timing.
This combines proven edges from research:
1. Vol spike reversion (ATR(7)/ATR(30) > 2.0) — captures panic exhaustion
2. Bollinger Band extremes (price < BB lower or > BB upper) — mean reversion trigger
3. HMA(50) trend filter — only trade with macro trend direction
4. RSI(14) confirmation — avoids catching falling knives too early

Key insight: Vol spikes precede reversals in crypto. After 2022 crash, vol spike + BB extreme
had 70%+ win rate on daily timeframe. HMA filter prevents counter-trend trades.

Changes from failed attempts:
- Simpler entry logic (no complex regime switching = more trades)
- Single HTF (4h) for trend confirmation, not multiple conflicting TFs
- Loose enough filters to generate 20-50 trades/year on 1d
- Position size 0.28 (discrete, conservative for 77% crash protection)

Target: Sharpe > 0.612 (beat current best), 25-40 trades/year, DD < -30%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_vol_spike_bb_reversion_hma_4h_atr_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
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

def calculate_bb(close, period=20, std_mult=2.0):
    """Bollinger Bands — mean reversion levels."""
    n = len(close)
    mid = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mid[i] = np.mean(window)
        std = np.std(window)
        upper[i] = mid[i] + std_mult * std
        lower[i] = mid[i] - std_mult * std
    
    return mid, upper, lower

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """
    ATR Ratio — measures volatility spike.
    Ratio > 2.0 indicates volatility expansion (panic/euphoria).
    Ratio < 1.2 indicates volatility contraction (calm).
    """
    n = len(close)
    atr_ratio = np.full(n, np.nan)
    
    atr_short = calculate_atr(high, low, close, period=short_period)
    atr_long = calculate_atr(high, low, close, period=long_period)
    
    for i in range(long_period, n):
        if atr_long[i] > 1e-10:
            atr_ratio[i] = atr_short[i] / atr_long[i]
    
    return atr_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h HMA for trend filter
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=50)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate primary (1d) indicators
    atr = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(high, low, close, short_period=7, long_period=30)
    bb_mid, bb_upper, bb_lower = calculate_bb(close, period=20, std_mult=2.0)
    rsi = calculate_rsi(close, period=14)
    hma_1d = calculate_hma(close, period=50)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(atr_ratio[i]):
            continue
        if np.isnan(bb_lower[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_1d[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === TREND FILTERS ===
        # 4h HMA for macro trend direction
        macro_bull = close[i] > hma_4h_aligned[i]
        macro_bear = close[i] < hma_4h_aligned[i]
        
        # 1d HMA for immediate trend
        trend_bull = close[i] > hma_1d[i]
        trend_bear = close[i] < hma_1d[i]
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 2.0
        vol_calm = atr_ratio[i] < 1.2
        
        # === BOLLINGER BAND EXTREMES ===
        bb_oversold = close[i] < bb_lower[i]
        bb_overbought = close[i] > bb_upper[i]
        bb_extreme = bb_oversold or bb_overbought
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Long entry: vol spike + BB oversold + RSI oversold + trend alignment
        if vol_spike and bb_oversold and rsi_oversold:
            if macro_bull or trend_bull:  # At least one trend filter bullish
                desired_signal = BASE_SIZE
        
        # Short entry: vol spike + BB overbought + RSI overbought + trend alignment
        if vol_spike and bb_overbought and rsi_overbought:
            if macro_bear or trend_bear:  # At least one trend filter bearish
                desired_signal = -BASE_SIZE
        
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