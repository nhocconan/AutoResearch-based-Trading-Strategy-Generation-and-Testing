#!/usr/bin/env python3
"""
Experiment #022: 4h Donchian + HMA Trend + Volume Spike + RSI Filter

HYPOTHESIS: Donchian(20) breakouts capture institutional momentum moves.
Combined with HMA(21) for trend direction (avoid counter-trend trades),
volume spike confirmation (>1.5x avg), and RSI filter (avoid neutral zones),
this creates tight entries that work in both bull (trend-follow longs)
and bear (counter-trend shorts at channel extremes).

WHY 4h: Best performer in DB. Camarilla+chop+vol = 1.47 Sharpe.
Why this combo: HMA Donchian + vol confirmed = 1.38 Sharpe.

TIMEFRAME: 4h
HTF: 12h for trend confirmation
TARGET: 100-250 total trades over 4 years
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_rsi_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - efficient vectorized version"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    # Use pandas for rolling WMA (much faster than loop)
    close_series = pd.Series(close)
    wma_half = close_series.rolling(half, min_periods=half).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    wma_full = close_series.rolling(period, min_periods=period).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    diff = 2.0 * wma_half - wma_full
    
    # Final WMA of diff
    diff_series = pd.Series(diff)
    result = diff_series.rolling(sqrt_n, min_periods=sqrt_n).apply(
        lambda x: np.sum(x * np.arange(1, len(x)+1)) / np.sum(np.arange(1, len(x)+1)), raw=True
    ).values
    
    return result

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - vectorized"""
    upper = pd.Series(high).rolling(period, min_periods=period).max().values
    lower = pd.Series(low).rolling(period, min_periods=period).min().values
    return upper, lower

def calculate_rsi(close, period=14):
    """RSI indicator"""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h HMA for trend alignment
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === CONDITIONS ===
        hma_trend_bull = close[i] > hma_12h_aligned[i]
        hma_trend_bear = close[i] < hma_12h_aligned[i]
        
        # Price relative to Donchian
        donch_mid = (donch_upper[i] + donch_lower[i]) / 2
        price_above_upper = close[i] > donch_upper[i]
        price_below_lower = close[i] < donch_lower[i]
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # RSI filter (avoid neutral zones)
        rsi_val = rsi_14[i]
        rsi_oversold = rsi_val < 35
        rsi_overbought = rsi_val > 65
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: Breakout above upper channel + bull trend + vol spike + RSI not overbought
            if price_above_upper and hma_trend_bull and vol_spike and rsi_val < 80:
                desired_signal = SIZE
            
            # SHORT: Breakdown below lower channel + bear trend + vol spike + RSI not oversold
            if price_below_lower and hma_trend_bear and vol_spike and rsi_val > 20:
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === RSI EXIT ===
        if in_position and position_side > 0:
            # Exit long on RSI overbought or return to lower channel
            if rsi_overbought and rsi_val > 75:
                desired_signal = 0.0
            if close[i] < donch_lower[i]:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short on RSI oversold or return to upper channel
            if rsi_oversold and rsi_val < 25:
                desired_signal = 0.0
            if close[i] > donch_upper[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
        
        signals[i] = desired_signal
    
    return signals