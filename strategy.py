#!/usr/bin/env python3
"""
Experiment #1448: 30m Primary + 4h/1d HTF — Ultra-Selective Confluence Strategy

Hypothesis: 30m strategies fail due to fee drag from too many trades. To succeed at 30m,
we must behave like a daily strategy — only trade when ALL confluence factors align.

Design (inspired by current best mtf_1d_donchian_hma_rsi_1w_atr_v1 Sharpe=0.618):
1. 4h HMA(21) = primary trend direction (call ONCE before loop)
2. 1d ATR percentile = volatility regime (low vol = trend, high vol = mean revert)
3. 30m RSI(14) extreme pullback = entry trigger (only <25 or >75)
4. Session filter = only 8-20 UTC (high liquidity, less noise)
5. Volume confirmation = >1.2x 20-bar average
6. Position size = 0.20 (conservative for lower TF)
7. Trailing stop = 2.5x ATR(14)

Why this might work:
- 4h trend filter prevents counter-trend trades (main failure mode)
- RSI extreme ensures we enter on pullbacks, not chases
- Session + volume filters eliminate 70%+ of bars from consideration
- Expected: 40-60 trades/year (within 30m limit of 50-100)

Target: Sharpe > 0.618, trades >= 30 train, >= 5 test, DD > -50%
Timeframe: 30m (MANDATORY per experiment instructions)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_4h_hma_1d_atr_rsi_extreme_session_vol_confluence_v1"
timeframe = "30m"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
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

def calculate_atr_percentile(atr, lookback=100):
    """ATR percentile within rolling lookback - measures vol regime"""
    n = len(atr)
    percentile = np.full(n, np.nan)
    
    for i in range(lookback, n):
        if not np.any(np.isnan(atr[i-lookback+1:i+1])):
            window = atr[i-lookback+1:i+1]
            rank = np.sum(window[:-1] < atr[i])
            percentile[i] = 100.0 * rank / (lookback - 1)
    
    return percentile

def calculate_volume_ma(volume, period=20):
    """Moving average of volume"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_ma

def get_utc_hour(open_time):
    """Extract UTC hour from open_time (milliseconds timestamp)"""
    # open_time is in milliseconds since epoch
    return (open_time // 3600000) % 24

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 4h HMA for trend direction
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 1d ATR percentile for vol regime
    atr_1d_raw = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=14)
    atr_1d_pct_raw = calculate_atr_percentile(atr_1d_raw, lookback=100)
    atr_1d_pct_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_pct_raw)
    
    # Calculate 30m indicators
    rsi_14 = calculate_rsi(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    vol_ma_20 = calculate_volume_ma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.20  # Conservative for 30m
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(atr_1d_pct_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        utc_hour = get_utc_hour(open_time[i])
        in_session = 8 <= utc_hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLUME FILTER (>1.2x average) ===
        volume_ratio = volume[i] / vol_ma_20[i] if vol_ma_20[i] > 1e-10 else 0.0
        high_volume = volume_ratio > 1.2
        
        # === 4H TREND DIRECTION ===
        trend_bull = close[i] > hma_4h_aligned[i]
        trend_bear = close[i] < hma_4h_aligned[i]
        
        # === 1D VOLATILITY REGIME ===
        # Low vol (<40th percentile) = trend following favored
        # High vol (>60th percentile) = mean reversion favored
        low_vol = atr_1d_pct_aligned[i] < 40.0
        high_vol = atr_1d_pct_aligned[i] > 60.0
        
        # === RSI EXTREME PULLBACK ===
        rsi_oversold = rsi_14[i] < 25.0
        rsi_overbought = rsi_14[i] > 75.0
        
        # === DESIRED SIGNAL - CONFLUENCE REQUIRED ===
        desired_signal = 0.0
        
        # LONG: 4h bull + (low vol + RSI pullback OR high vol + extreme oversold) + volume
        if trend_bull and high_volume:
            if low_vol and rsi_14[i] < 40.0:  # Pullback in low vol trend
                desired_signal = BASE_SIZE
            elif high_vol and rsi_oversold:  # Extreme oversold in high vol
                desired_signal = BASE_SIZE
        
        # SHORT: 4h bear + (low vol + RSI pullback OR high vol + extreme overbought) + volume
        elif trend_bear and high_volume:
            if low_vol and rsi_14[i] > 60.0:  # Pullback in low vol trend
                desired_signal = -BASE_SIZE
            elif high_vol and rsi_overbought:  # Extreme overbought in high vol
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
        if abs(desired_signal) >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                # Flip position
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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