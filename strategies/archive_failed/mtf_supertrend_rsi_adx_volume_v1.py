#!/usr/bin/env python3
"""
EXPERIMENT #018 - Supertrend + RSI Pullback + ADX Strength + Volume Filter
===============================================================================
Hypothesis: Supertrend provides cleaner trend signals than Donchian channels.
Combined with RSI pullback entries in the direction of the 4h trend, ADX strength
filter (>20) to avoid choppy markets, and volume confirmation for conviction.
This should reduce whipsaws while capturing major trending moves.

Why this might beat Sharpe=2.139:
- Supertrend (tested in #002, #011) showed strong returns (+433%, +497%)
- RSI pullback entries avoid chasing extended moves
- ADX filter removes low-quality choppy periods
- Volume confirmation adds conviction to breakouts
- Simpler calculations than Donchian+HMA combo (avoids #007 timeout)
- Discrete position sizing (0.0, ±0.25, ±0.35) reduces churn costs
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_rsi_adx_volume_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n)  # 1 for bullish, -1 for bearish
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band[i] = hl2 + multiplier * atr[i]
        lower_band[i] = hl2 - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            direction[i] = 1
        else:
            if close[i - 1] <= supertrend[i - 1]:
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
                if close[i] > supertrend[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    direction[i] = -1
            else:
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
                if close[i] < supertrend[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
                else:
                    direction[i] = 1
    
    return supertrend, direction


def calculate_rsi(close, period=14):
    """Calculate RSI with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    mask = (avg_loss > 0) & (avg_gain >= 0)
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i - 1]) if (high[i] - high[i - 1]) > (low[i - 1] - low[i]) else 0
        minus_dm[i] = max(0, low[i - 1] - low[i]) if (low[i - 1] - low[i]) > (high[i] - high[i - 1]) else 0
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
    
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / atr[mask]
    
    dx = np.zeros(n)
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    
    return adx


def calculate_volume_sma(volume, period=20):
    """Calculate volume SMA for confirmation"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices.get("volume", np.ones(len(close))).values
    n = len(close)
    
    # 1h indicators for entry timing
    rsi_1h = calculate_rsi(close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    adx_1h = calculate_adx(high, low, close, period=14)
    vol_sma_1h = calculate_volume_sma(volume, period=20)
    
    # Resample to 4h for trend filter
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # 4h Supertrend for trend direction
    supertrend_4h, direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(direction_4h) and idx_4h >= 10:
            trend_1h[i] = direction_4h[idx_4h]
    
    # Generate signals
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels
    SIZE_FULL = 0.35
    SIZE_HALF = 0.25
    
    # Thresholds
    RSI_LONG_ENTRY = 40
    RSI_SHORT_ENTRY = 60
    ADX_MIN = 20
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 28, 20)
    
    entry_price = np.zeros(n)
    position_side = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(rsi_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        rsi_val = rsi_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        vol = volume[i]
        vol_avg = vol_sma_1h[i]
        
        # ATR filter - avoid extreme volatility
        if atr > 0 and atr / price > 0.05:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
        
        # ADX filter - only trade when trend is strong enough
        if adx_val < ADX_MIN:
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        # Volume confirmation
        volume_confirmed = vol > vol_avg * 0.8
        
        if trend == 1:  # 4h uptrend
            if rsi_val < RSI_LONG_ENTRY and volume_confirmed:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
            elif rsi_val < 50 and volume_confirmed:
                signals[i] = SIZE_HALF
                position_side[i] = 1
                entry_price[i] = price
            else:
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            if rsi_val > RSI_SHORT_ENTRY and volume_confirmed:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
            elif rsi_val > 50 and volume_confirmed:
                signals[i] = -SIZE_HALF
                position_side[i] = -1
                entry_price[i] = price
            else:
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
        else:
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals