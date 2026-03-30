#!/usr/bin/env python3
"""
Experiment #007: 6h ATR Band Breakout + 1d ADX Trend + Volume

HYPOTHESIS: ATR bands adapt to volatility regime better than fixed-price channels.
Band width automatically adjusts to current market conditions.

WHY 6h:
- 12h has 54% keep rate but few total strategies tested
- 6h is challenging (23% keep) but offers different signal profile
- ATR bands on 6h capture multi-day volatility cycles

KEY MECHANISM:
- ATR(14) band around EMA(21) centerline
- 2.5x multiplier for band width (adjusts dynamically to volatility)
- Entry when price closes beyond band + 1d ADX trend confirmation + volume spike
- This should generate 20-35 trades/year with quality filtering

TARGET: 80-140 total trades over 4 years (20-35/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_atr_band_adx_vol_1d_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_adx(high, low, close, period=14):
    """Directional Movement Index - measures trend strength, not direction"""
    n = len(close)
    if n < period * 2 + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth using EMA
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI values
    plus_di = np.where(atr_smooth > 0, 100 * plus_dm_smooth / atr_smooth, 0)
    minus_di = np.where(atr_smooth > 0, 100 * minus_dm_smooth / atr_smooth, 0)
    
    # DX
    dx = np.where((plus_di + minus_di) > 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX = EMA of DX
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX for structural trend strength
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, period=20)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # EMA centerline for ATR bands
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # ATR band upper/lower around EMA
    atr_multiplier = 2.5
    upper_band = ema_21 + atr_multiplier * atr_14
    lower_band = ema_21 - atr_multiplier * atr_14
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Generate signals ===
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # 21 EMA + 14 ATR + 20 vol MA + 1d alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # === CONDITIONS ===
        # 1d ADX > 25 = trending (filter out range markets)
        adx_trending = adx_aligned[i] > 25
        
        # Volume spike confirmation (1.8x average)
        vol_spike = vol_ratio[i] > 1.8
        
        # ATR band breakout
        # Long: price breaks above upper band
        # Short: price breaks below lower band
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout above upper band + trending + volume spike
            if breakout_up and adx_trending and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Breakout below lower band + trending + volume spike
            if breakout_down and adx_trending and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (trailing ATR stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2.5 ATR from recent high
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if ADX drops below 20 (trend weakening)
                if adx_aligned[i] < 20:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2.5 ATR from recent low
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if ADX drops below 20 (trend weakening)
                if adx_aligned[i] < 20:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals