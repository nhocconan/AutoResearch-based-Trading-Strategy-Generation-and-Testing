#!/usr/bin/env python3
"""
Experiment #095: 6h Primary + 12h/1d HTF — Weekly Trend + Daily HMA + RSI Stochastic + Volume

Hypothesis: 6h timeframe sits between 4h (too noisy) and 12h (too slow). Key insight from 89 failed experiments:
- Weekly trend bias is more stable than daily for major direction
- Daily HMA confirms intermediate trend without whipsaw
- 6h RSI + Stochastic confluence provides entry timing
- Volume confirmation filters false breakouts (missing in most failed 6h strategies)
- ATR volatility filter avoids entering during panic spikes
- LOOSE entry thresholds to ensure >=30 trades on train, >=3 on test

Key design choices:
- Timeframe: 6h (30-60 trades/year target)
- HTF: 1w HMA for major trend, 1d HMA for intermediate confirmation
- Entry: RSI(14) + Stoch(14,3,3) confluence + volume spike confirmation
- Regime: Weekly HMA slope determines long/short bias
- Position size: 0.28 (28% of capital, conservative for 6h)
- Stoploss: 2.5x ATR trailing
- LOOSE filters: RSI 25-75, Stoch 20-80 to ensure trade generation

Target: Sharpe>0.167 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test, ALL symbols Sharpe>0
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_weekly_trend_daily_hma_rsi_stoch_vol_v1"
timeframe = "6h"
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

def calculate_stochastic(high, low, close, k_period=14, d_period=3):
    """Stochastic Oscillator %K and %D"""
    n = len(close)
    if n < k_period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    k = np.zeros(n)
    k[:] = np.nan
    
    for i in range(k_period - 1, n):
        lowest_low = np.min(low[i - k_period + 1:i + 1])
        highest_high = np.max(high[i - k_period + 1:i + 1])
        range_hl = highest_high - lowest_low
        
        if range_hl > 1e-10:
            k[i] = 100.0 * (close[i] - lowest_low) / range_hl
        else:
            k[i] = 50.0
    
    d = pd.Series(k).ewm(span=d_period, min_periods=d_period, adjust=False).mean().values
    
    return k, d

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss and volatility filter"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_volume_ratio(volume, period=20):
    """Volume ratio vs moving average - detects volume spikes"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / (vol_ma + 1e-10)
    
    return vol_ratio

def calculate_hma_slope(hma, lookback=5):
    """Calculate HMA slope over lookback period"""
    n = len(hma)
    slope = np.zeros(n)
    slope[:] = np.nan
    
    for i in range(lookback, n):
        if not np.isnan(hma[i]) and not np.isnan(hma[i - lookback]):
            slope[i] = (hma[i] - hma[i - lookback]) / (hma[i - lookback] + 1e-10)
    
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 1w HMA slope for trend direction
    hma_1w_slope_raw = calculate_hma_slope(hma_1w_raw, lookback=3)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope_raw)
    
    # Calculate and align 1d HMA for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=34)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    hma_6h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    stoch_k, stoch_d = calculate_stochastic(high, low, close, k_period=14, d_period=3)
    atr = calculate_atr(high, low, close, period=14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size (conservative for 6h)
    
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
        if np.isnan(hma_6h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]):
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
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (1w HMA slope for major trend) ===
        weekly_bull = hma_1w_slope_aligned[i] > 0.001  # positive slope
        weekly_bear = hma_1w_slope_aligned[i] < -0.001  # negative slope
        weekly_neutral = not weekly_bull and not weekly_bear
        
        # Price above/below weekly HMA
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_aligned[i]
        
        # === INTERMEDIATE TREND (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_aligned[i]
        
        # === 6h HMA TREND ===
        hma_6h_bull = close[i] > hma_6h[i]
        hma_6h_bear = close[i] < hma_6h[i]
        
        # === RSI SIGNALS (LOOSE thresholds for trade generation) ===
        rsi_oversold = rsi[i] < 45.0  # loose for longs
        rsi_overbought = rsi[i] > 55.0  # loose for shorts
        rsi_neutral = 35.0 <= rsi[i] <= 65.0
        
        # === STOCHASTIC SIGNALS ===
        stoch_oversold = stoch_k[i] < 40.0 and stoch_d[i] < 40.0
        stoch_overbought = stoch_k[i] > 60.0 and stoch_d[i] > 60.0
        stoch_bull_cross = stoch_k[i] > stoch_d[i] and stoch_k[i-1] <= stoch_d[i-1] if i > 0 else False
        stoch_bear_cross = stoch_k[i] < stoch_d[i] and stoch_k[i-1] >= stoch_d[i-1] if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirmed = vol_ratio[i] > 0.8  # at least 80% of avg volume (not dried up)
        vol_spike = vol_ratio[i] > 1.5  # volume spike for breakout confirmation
        
        # === VOLATILITY FILTER (avoid panic entries) ===
        # ATR ratio: current ATR vs 50-bar average ATR
        atr_50 = pd.Series(atr[:i+1]).rolling(window=50, min_periods=50).mean().iloc[-1] if i >= 49 else atr[i]
        vol_filter = not (np.isnan(atr_50) or atr[i] > 3.0 * atr_50)  # avoid extreme vol spikes
        
        # === DESIRED SIGNAL (Multi-confluence logic) ===
        desired_signal = 0.0
        
        # LONG CONDITIONS (Weekly bull + Daily confirm + 6h entry)
        long_conditions = (
            (weekly_bull or price_above_1w_hma) and  # Weekly trend or price above
            price_above_1d_hma and  # Daily confirm
            hma_6h_bull and  # 6h trend
            rsi_oversold and  # RSI not overbought
            vol_confirmed and  # Volume ok
            vol_filter  # Not extreme volatility
        )
        
        # Strong long: add stochastic confirmation
        strong_long = long_conditions and (stoch_oversold or stoch_bull_cross)
        
        # SHORT CONDITIONS (Weekly bear + Daily confirm + 6h entry)
        short_conditions = (
            (weekly_bear or price_below_1w_hma) and  # Weekly trend or price below
            price_below_1d_hma and  # Daily confirm
            hma_6h_bear and  # 6h trend
            rsi_overbought and  # RSI not oversold
            vol_confirmed and  # Volume ok
            vol_filter  # Not extreme volatility
        )
        
        # Strong short: add stochastic confirmation
        strong_short = short_conditions and (stoch_overbought or stoch_bear_cross)
        
        # === ASSIGN SIGNALS ===
        if strong_long:
            desired_signal = SIZE
        elif long_conditions:
            desired_signal = SIZE * 0.7  # weaker long
        elif strong_short:
            desired_signal = -SIZE
        elif short_conditions:
            desired_signal = -SIZE * 0.7  # weaker short
        
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