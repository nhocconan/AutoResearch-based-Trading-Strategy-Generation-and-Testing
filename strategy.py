#!/usr/bin/env python3
"""
Experiment #007: 6h Bollinger Squeeze Breakout + 1d SMA200 Regime

HYPOTHESIS: Bollinger Band Width contracts to 63-bar minimum = institutional
accumulation/distribution. Subsequent band breakout with volume confirmation
captures the explosive move that follows. 1d SMA200 regime ensures entries
only in direction of major trend.

WHY 6h: Fewer bars = squeeze conditions less frequent = tighter entries.
1d SMA200 filter prevents bear market whipsaws (BTC crashed 77% in 2022).

TARGET: 50-100 total trades over 4 years (12-25/year). Very tight entries.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_bb_squeeze_vol_sma200_1d_v1"
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

def calculate_bb_width_pct(close, period=20, mult=2.0):
    """Bollinger Band Width as percentage of SMA (normalized for comparison)"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # Standard Bollinger Bands
    close_ma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    close_std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper_band = close_ma + mult * close_std
    lower_band = close_ma - mult * close_std
    
    # Band width as percentage of middle band (normalized)
    bb_width = (upper_band - lower_band) / close_ma * 100
    
    return bb_width, close_ma

def calculate_vol_ma(volume, period=20):
    """Volume moving average"""
    return pd.Series(volume).rolling(window=period, min_periods=period).mean().values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    bb_width, close_ma = calculate_bb_width_pct(close, period=20, mult=2.0)
    vol_ma = calculate_vol_ma(volume, period=20)
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Rolling minimum of BB width (63 bars = ~15 days at 6h)
    # Lower quartile check for squeeze
    bb_width_roll63 = pd.Series(bb_width).rolling(window=63, min_periods=63).min().values
    bb_width_q25 = pd.Series(bb_width).rolling(window=63, min_periods=30).quantile(0.25).values
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    # Warmup: need 200 for SMA200, 63 for BB width min
    warmup = 200
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width_roll63[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Bollinger parameters
        sma20 = close_ma[i]
        upper_band = sma20 + 2.0 * pd.Series(close).rolling(20, min_periods=20).std().iloc[i] if i >= 20 else np.nan
        lower_band = sma20 - 2.0 * pd.Series(close).rolling(20, min_periods=20).std().iloc[i] if i >= 20 else np.nan
        
        # Recalculate bands properly
        if i >= 20:
            bb_std = pd.Series(close[i-20:i+1]).std()
            upper_band = sma20 + 2.0 * bb_std
            lower_band = sma20 - 2.0 * bb_std
        else:
            upper_band = lower_band = np.nan
        
        # === REGIME CHECK (1d SMA200) ===
        price_above_sma = close[i] > sma_1d_aligned[i]
        
        # === SQUEEZE DETECTION ===
        # BB width at or near 63-bar minimum = squeeze
        bb_min = bb_width_roll63[i]
        is_squeeze = bb_width[i] <= bb_min * 1.05  # within 5% of minimum
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Squeeze breakout LONG: above SMA200, BB squeeze active, break upper band, volume confirm
            if price_above_sma and is_squeeze and vol_spike:
                if close[i] > upper_band:
                    desired_signal = SIZE
            
            # Squeeze breakout SHORT: below SMA200, BB squeeze active, break lower band, volume confirm
            if not price_above_sma and is_squeeze and vol_spike:
                if close[i] < lower_band:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR) ===
        if in_position and position_side > 0:
            stop_price = entry_price - 2.0 * entry_atr
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            stop_price = entry_price + 2.0 * entry_atr
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars (~1 day) to avoid whipsaw ===
        bars_held = i - entry_bar
        if in_position and bars_held < 4:
            # Don't exit early, maintain position
            desired_signal = position_side * SIZE
        
        # === TAKE PROFIT: 2:1 RR ===
        if in_position and bars_held >= 4:
            if position_side > 0:
                profit_target = entry_price + 2.0 * entry_atr
                if high[i] >= profit_target:
                    desired_signal = 0.0  # Exit on profit target
            if position_side < 0:
                profit_target = entry_price - 2.0 * entry_atr
                if low[i] <= profit_target:
                    desired_signal = 0.0  # Exit on profit target
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals