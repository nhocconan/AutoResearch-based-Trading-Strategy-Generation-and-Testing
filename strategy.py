#!/usr/bin/env python3
"""
Experiment #007: 6h Ichimoku Cloud + Weekly EMA21 + Volume

HYPOTHESIS: Ichimoku's TK cross (Tenkan-Kijun) captures momentum shifts,
the Cloud (Senkou B) provides dynamic support/resistance, and weekly EMA21
ensures entries align with the larger trend.

WHY IT'S DIFFERENT:
- Uses TK cross as MEAN REVERSION signal (cross below -50 = oversold, 
  cross above +50 = overbought)
- Cloud as CONFIRMATION (price must be near cloud for valid signal)
- NOT a trend-following system — fades extremes with trend alignment

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- TK cross works in any market phase (momentum oscillates)
- Weekly EMA filters direction without being too slow
- Volume confirmation on cross reduces false signals

TARGET: 75-150 total trades over 4 years = 19-37/year.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_ichimoku_cloud_ema21_1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ichimoku(high, low, close, period_fast=9, period_med=26, period_slow=52):
    """
    Calculate Ichimoku Cloud components.
    Returns: tenkan, kijun, senkou_a, senkou_b, chikou
    """
    n = len(close)
    
    # Tenkan-sen (Conversion Line): (9-period highest + 9-period lowest) / 2
    tenkan = np.zeros(n)
    for i in range(n):
        start = max(0, i - period_fast + 1)
        tenkan[i] = (np.max(high[start:i+1]) + np.min(low[start:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period highest + 26-period lowest) / 2
    kijun = np.zeros(n)
    for i in range(n):
        start = max(0, i - period_med + 1)
        kijun[i] = (np.max(high[start:i+1]) + np.min(low[start:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward 26 periods
    senkou_a_raw = (tenkan + kijun) / 2
    
    # Senkou Span B (Leading Span B): (52-period highest + 52-period lowest) / 2, shifted 26
    senkou_b_raw = np.zeros(n)
    for i in range(n):
        start = max(0, i - period_slow + 1)
        senkou_b_raw[i] = (np.max(high[start:i+1]) + np.min(low[start:i+1])) / 2
    
    # Chikou Span (Lagging Span): current close, shifted back 26
    chikou = close.copy()
    
    # Pad Senkou A and B for forward shift (26 bars)
    padding = np.full(period_med, np.nan)
    senkou_a = np.concatenate([padding, senkou_a_raw[:n-period_med]])
    senkou_b = np.concatenate([padding, senkou_b_raw[:n-period_med]])
    
    return tenkan, kijun, senkou_a, senkou_b, chikou

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load weekly HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Calculate Ichimoku on 6h (local) ===
    tenkan, kijun, senkou_a, senkou_b, chikou = calculate_ichimoku(high, low, close)
    
    # === Local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
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
    entry_bar = 0
    
    warmup = max(100, 52 + 26 + 10)  # Need 52 for Ichimoku + buffer
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if weekly EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === WEEKLY TREND (1w EMA21) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ICHIMOKU SIGNALS ===
        # TK cross value (normalized by recent range for comparison)
        tk_diff = tenkan[i] - kijun[i]
        tk_range = kijun[i] - low[i] if kijun[i] > tenkan[i] else high[i] - kijun[i]
        tk_range = max(tk_range, close[i] * 0.001)  # Avoid division by zero
        
        # Cloud boundaries (current values for comparison)
        cloud_top = max(senkou_a[i], senkou_b[i])
        cloud_bottom = min(senkou_a[i], senkou_b[i])
        cloud_thickness = cloud_top - cloud_bottom
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TK crosses UP through Kijun + price near/behind cloud + weekly trend up ===
            # TK cross up means Tenkan crosses above Kijun
            if price_above_1w_ema and vol_spike:
                # Tenkan just crossed above Kijun (check prev bar)
                if tenkan[i-1] <= kijun[i-1] and tenkan[i] > kijun[i]:
                    # Price is near cloud bottom (within 1.5x ATR) or above it
                    if close[i] <= cloud_top + 1.5 * atr_14[i]:
                        desired_signal = SIZE
            
            # === SHORT: TK crosses DOWN through Kijun + price near/behind cloud + weekly trend down ===
            if not price_above_1w_ema and vol_spike:
                if tenkan[i-1] >= kijun[i-1] and tenkan[i] < kijun[i]:
                    if close[i] >= cloud_bottom - 1.5 * atr_14[i]:
                        desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
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
        
        # === HOLD PERIOD (minimum 2 bars = 12h to avoid churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 2:
            # Take profit on trend reversion
            if position_side > 0 and close[i] < ema_1w_aligned[i]:
                desired_signal = 0.0
            if position_side < 0 and close[i] > ema_1w_aligned[i]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals