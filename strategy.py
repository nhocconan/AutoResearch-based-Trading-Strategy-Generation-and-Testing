#!/usr/bin/env python3
"""
Experiment #006: 4h Donchian(20) + TRIX(14) + Volume Confirmation

HYPOTHESIS: Donchian(20) breakout on 4h captures major trend shifts.
TRIX provides smooth momentum without noise. Combining both with volume
confirms institutional involvement. HTF 1d EMA50 filters direction.

WHY IT SHOULD WORK: Donchian breakout catches sustained moves after
consolidation. TRIX eliminates false signals by requiring momentum.
Volume confirms institutional conviction. HTF trend alignment prevents
trading against the primary trend.

TRADE COUNT ESTIMATE: ~2-3 breakouts/month/symbol = 24-36/year = 96-144 over 4 years.
This is within target range (75-200). TRIX filter may reduce to 75-100.

SIGNAL SIZE: 0.30 (discrete levels only).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_trix_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(prices, period=14):
    """TRIX indicator - triple smoothed EMA rate of change"""
    ema1 = pd.Series(prices).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    trix = 100 * ema3.pct_change(period)
    return trix.values

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
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix_14 = calculate_trix(close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Donchian channels (20 periods)
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_upper + donch_lower) / 2
    
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
    prev_signal = 0.0
    
    warmup = 50  # Need enough for Donchian(20) + TRIX alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(trix_14[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # TRIX momentum (previous bar for signal stability)
        trix_positive = trix_14[i] > 0 and trix_14[i-1] <= 0 if i > warmup else trix_14[i] > 0
        trix_negative = trix_14[i] < 0 and trix_14[i-1] >= 0 if i > warmup else trix_14[i] < 0
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.4
        
        # === DONCHIAN BREAKOUT (use closed bars for signal bar) ===
        # Long: price breaks above 20-high with momentum
        # Short: price breaks below 20-low with momentum
        
        # Previous bar values for confirmation
        prev_donch_upper = donch_upper[i - 1] if i > warmup else donch_upper[i]
        prev_donch_lower = donch_lower[i - 1] if i > warmup else donch_lower[i]
        prev_close = close[i - 1]
        prev_trix = trix_14[i - 1] if i > warmup else trix_14[i]
        
        # === ENTRY LOGIC ===
        desired_signal = prev_signal  # Default: hold current position
        
        if not in_position:
            desired_signal = 0.0
            
            # === LONG: Breakout above Donchian upper with TRIX turning positive ===
            if price_above_1d_ema and trix_positive and vol_spike:
                # Check if previous bar closed below/broke upper band (confirms breakout)
                if prev_close <= prev_donch_upper or high[i] > prev_donch_upper:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian lower with TRIX turning negative ===
            if not price_above_1d_ema and trix_negative and vol_spike:
                if prev_close >= prev_donch_lower or low[i] < prev_donch_lower:
                    desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.0 ATR) ===
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
        
        # === TIME EXIT (hold at least 8 bars = 2 days to avoid churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 8:
            # Exit on opposite TRIX signal or Donchian reversal
            if position_side > 0 and trix_negative:
                desired_signal = 0.0
            if position_side < 0 and trix_positive:
                desired_signal = 0.0
        
        # === DONCHIAN REVERSAL EXIT ===
        if in_position and bars_held >= 4:
            if position_side > 0 and low[i] < donch_lower[i - 1]:
                desired_signal = 0.0
            if position_side < 0 and high[i] > donch_upper[i - 1]:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != prev_signal:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                if desired_signal != 0.0:
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
                    in_position = False
                    position_side = 0
                    stop_price = 0.0
        
        signals[i] = desired_signal
        prev_signal = desired_signal
    
    return signals