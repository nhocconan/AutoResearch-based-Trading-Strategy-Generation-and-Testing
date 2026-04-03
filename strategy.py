#!/usr/bin/env python3
"""
Experiment #2214: 1h Donchian(20) breakout + 4h/1d trend filter + volume confirmation + session filter
HYPOTHESIS: 1h timeframe with 4h/1d HTF trend filters captures swing momentum while minimizing fee drag.
- Primary: 1h Donchian(20) breakout with volume > 1.5x 20-bar average
- HTF: 4h HMA(21) and 1d EMA(50) trend filters (only trade when both align)
- Session: Trade only 08:00-20:00 UTC to avoid low-volume Asian session noise
- Stoploss: ATR(14) trailing stop (2*ATR) or opposite Donchian touch
- Target: 60-150 total trades over 4 years (15-37/year) - optimized for 1h timeframe
- Designed to work in bull markets (trend following) and bear markets (mean reversion at extremes)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2214_1h_donchian20_4h_1d_trend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h data for HMA trend (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h HMA(21): Hull Moving Average
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    # Calculate WMA for close_4h
    wma_full = np.array([np.nan] * len(close_4h))
    wma_half = np.array([np.nan] * len(close_4h))
    
    for i in range(20, len(close_4h)):  # 21-1 = 20 for WMA(21)
        wma_full[i] = np.mean(close_4h[i-20:i+1] * np.arange(1, 22))
    for i in range(half_len-1, len(close_4h)):
        wma_half[i] = np.mean(close_4h[i-half_len+1:i+1] * np.arange(1, half_len+1))
    
    # HMA = WMA(2*WMA_half - WMA_full, sqrt_len)
    wma_diff = 2 * wma_half - wma_full
    hma_4h = np.array([np.nan] * len(close_4h))
    for i in range(sqrt_len-1, len(close_4h)):
        if i >= half_len-1 and not np.isnan(wma_diff[i]):
            hma_4h[i] = np.mean(wma_diff[i-sqrt_len+1:i+1] * np.arange(1, sqrt_len+1))
    
    # Trend: 1 if close > HMA, -1 otherwise
    trend_4h = np.where(close_4h > hma_4h, 1, -1)
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 1h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size - conservative for risk management
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_4h_aligned[i]) or np.isnan(trend_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hour = hours[i]
        
        # --- Session Filter ---
        if not (8 <= hour <= 20):  # Trade only 08:00-20:00 UTC
            signals[i] = 0.0
            continue
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require both 4h and 1d trend alignment for bias filter
        trend_bias_4h = trend_4h_aligned[i]
        trend_bias_1d = trend_1d_aligned[i]
        
        # Only trade when both HTF trends agree
        if trend_bias_4h == trend_bias_1d:
            # Volume confirmation: require volume spike (> 1.5x average)
            volume_spike = vol_ratio[i] > 1.5
            
            if volume_spike:
                # Long entry: price breaks above upper Donchian AND trends up
                if trend_bias_4h > 0 and price > donchian_upper[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = SIZE
                # Short entry: price breaks below lower Donchian AND trends down
                elif trend_bias_4h < 0 and price < donchian_lower[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    highest_since_entry = high[i]
                    lowest_since_entry = low[i]
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals