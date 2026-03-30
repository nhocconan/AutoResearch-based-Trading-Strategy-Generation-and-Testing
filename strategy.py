#!/usr/bin/env python3
"""
Experiment #025: 6h Volatility Compression Breakout + Daily ATR Regime

HYPOTHESIS: Markets alternate between volatility compression (consolidation) and expansion.
When BB Width contracts to 30d low AND price breaks compression with volume, 
explosive moves follow. Daily ATR regime filter prevents trading during 
high-volatility panic (2022 crash, black swan events).

WHY IT WORKS IN BOTH MARKETS:
- Bull: Compression under BB + breakout up + ATR regime neutral → ride momentum
- Bear: Same signal but ATR regime filters out crash periods (random direction)
- Range expansion follows compression (physics of markets)

EXPECTED TRADES: 75-150 total over 4 years (19-37/year)
- BB Width at 30d low ≈ 5-10% of bars
- Volume confirmation reduces by ~40%
- Daily ATR filter reduces by ~30%
- Final: ~75-120 trades per symbol

DIFFERENT FROM DONCHIAN: Uses BB Width %B for compression detection, 
not just price channel breakouts. Captures "coil" before explosive move.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vol_compression_breakout_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bb_width(high, low, close, period=20):
    """Bollinger Band Width as % of price"""
    mid = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = mid + 2 * std
    lower = mid - 2 * std
    
    width = (upper - lower) / mid
    return width

def calculate_atr_ratio(atr, period=14):
    """ATR ratio: current ATR / 30-bar ATR MA. >1.5 = high volatility regime."""
    atr_ma = pd.Series(atr).rolling(window=30, min_periods=30).mean().values
    ratio = atr / np.where(atr_ma > 0, atr_ma, 1)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ATR for regime
    daily_atr = calculate_atr(
        df_1d['high'].values,
        df_1d['low'].values,
        df_1d['close'].values,
        period=14
    )
    daily_atr_aligned = align_htf_to_ltf(prices, df_1d, daily_atr)
    
    # Daily EMA(20) for trend
    daily_ema = pd.Series(df_1d['close'].values).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema_aligned = align_htf_to_ltf(prices, df_1d, daily_ema)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_ratio = calculate_atr_ratio(atr_14)
    
    # BB Width for compression detection
    bb_width = calculate_bb_width(high, low, close, period=20)
    
    # Rolling min/max of BB Width (30 bars) for compression detection
    bb_width_min = pd.Series(bb_width).rolling(window=30, min_periods=30).min().values
    bb_width_max = pd.Series(bb_width).rolling(window=30, min_periods=30).max().values
    
    # BB for price reference
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Price momentum (5 bars)
    roc = pd.Series(close).pct_change(5).values
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Enough for BB20, ATR14, 30-bar lookback
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width_min[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(daily_atr_aligned[i]) or np.isnan(daily_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER: Daily ATR ratio ===
        # Skip if daily ATR ratio > 1.8 (high volatility regime = choppy)
        daily_atr_ratio = calculate_atr_ratio(daily_atr_aligned)
        if not np.isnan(daily_atr_ratio[i]) and daily_atr_ratio[i] > 1.8:
            # High volatility - stay flat
            if not in_position:
                signals[i] = 0.0
                continue
        
        # === COMPRESSION DETECTION ===
        # BB Width at 30d low = volatility compression
        width_range = bb_width_max[i] - bb_width_min[i]
        if width_range > 1e-10:
            width_percentile = (bb_width[i] - bb_width_min[i]) / width_range
        else:
            width_percentile = 1.0
        
        is_compressed = width_percentile < 0.15  # Bottom 15% of 30d range
        
        # === BREAKOUT DETECTION ===
        # Price breaks above BB mid with momentum after compression
        bull_breakout = (close[i] > bb_mid[i] and roc[i] > 0.01)
        bear_breakout = (close[i] < bb_mid[i] and roc[i] < -0.01)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === TREND: Daily EMA alignment ===
        bull_trend = close[i] > daily_ema_aligned[i]
        bear_trend = close[i] < daily_ema_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Compression + breakout + volume + bull trend
            if is_compressed and bull_breakout and vol_spike and bull_trend:
                desired_signal = SIZE
            
            # SHORT: Compression + breakdown + volume + bear trend
            elif is_compressed and bear_breakout and vol_spike and bear_trend:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if daily trend flips
                elif close[i] < daily_ema_aligned[i] * 0.995:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if daily trend flips
                elif close[i] > daily_ema_aligned[i] * 1.005:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals