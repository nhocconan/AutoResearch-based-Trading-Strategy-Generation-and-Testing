#!/usr/bin/env python3
"""
Experiment #022: 12h Donchian + Volume Spike + 1d HMA Trend

HYPOTHESIS: 12h Donchian(20) breakouts mark institutional accumulation/distribution
points. Volume spike confirms the move isn't false. 1d HMA(21) provides trend bias
to filter counter-trend entries. This combination worked in DB (test_sharpe=1.38-1.46)
when entry conditions were strict. 12h is slow enough to avoid overtrading while
capturing major trend moves. Works in both bull and bear via long breakouts and
short breakdowns when trend aligned.

TIMEFRAME: 12h primary | HTF: 1d for trend
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_1d_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - vectorized for speed"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = pd.Series(series).rolling(window=span, min_periods=span).apply(
            lambda x: np.sum(x * weights[:len(x)]) / np.sum(weights[:len(x)]), raw=True
        ).values
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.where(
        np.isnan(wma_half) | np.isnan(wma_full),
        np.nan,
        2.0 * wma_half - wma_full
    )
    
    return wma(diff, sqrt_n)

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
    """Donchian Channel - upper and lower bands"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d HMA for trend bias - aligned to 12h
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    donch_upper, donch_lower = calculate_donchian(high, low, period=20)
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30  # discrete position size
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]):
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d_hma = close[i] > hma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION (strict: 1.5x) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN CHANNEL STATE ===
        channel_width = donch_upper[i] - donch_lower[i]
        price_vs_upper = close[i] > donch_upper[i]
        price_vs_lower = close[i] < donch_lower[i]
        
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # Price breaks above upper band + volume spike + bullish trend
            if price_vs_upper and vol_spike and price_above_1d_hma:
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # Price breaks below lower band + volume spike + bearish trend
            if price_vs_lower and vol_spike and not price_above_1d_hma:
                desired_signal = -SIZE
        
        else:
            # === HOLDING POSITION ===
            # ATR-based stoploss (2.5x)
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                else:
                    desired_signal = SIZE
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                else:
                    desired_signal = -SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction change
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals