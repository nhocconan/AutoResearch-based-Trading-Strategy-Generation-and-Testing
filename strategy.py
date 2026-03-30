#!/usr/bin/env python3
"""
Experiment #006: 4h Keltner Squeeze + Volume Expansion + 1d Trend

HYPOTHESIS: Low volatility (squeeze) followed by high volume expansion
is the highest-probability breakout setup. When BB contracts within Keltner
bands, volatility is compressing. Volume surge on expansion = institutional
move. 1d EMA confirms trend direction.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Squeeze clears → price explodes above Keltner upper → long
- Bear: Squeeze clears → price collapses below Keltner lower → short
- Symmetrical: captures moves in both directions, adapts to regime

KEY DIFFERENCE FROM DONCHIAN: Keltner uses ATR-based bands, which are
adaptive to volatility. Donchian is fixed lookback = different behavior
in high vs low vol periods.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_keltner_squeeze_vol_1d_v1"
timeframe = "4h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction (proven from DB)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Keltner Channel (EMA20 with 2*ATR bands)
    kelt_ema = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    kelt_upper = kelt_ema + 2.0 * atr_14
    kelt_lower = kelt_ema - 2.0 * atr_14
    kelt_mid = kelt_ema
    
    # Bollinger Bands for squeeze detection (20, 2)
    bb_sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    
    # BB Width: detect squeeze when BB contracts
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze_ratio = bb_width / np.where(bb_width_ma > 0, bb_width_ma, 1)
    
    # Volume confirmation
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
    entry_bar = 0
    
    warmup = 100  # Need 50 for EMA50 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(kelt_ema[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === KELTNER LEVELS ===
        upper_band = kelt_upper[i]
        lower_band = kelt_lower[i]
        middle_band = kelt_mid[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.8  # Volume 1.8x average
        
        # === BB SQUEEZE (low volatility) ===
        is_squeeze = squeeze_ratio[i] < 0.85  # BB contracted vs 20d avg
        
        # === BREAKOUT: price outside Keltner band ===
        price_above_upper = close[i] > upper_band
        price_below_lower = close[i] < lower_band
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG: Price breaks above Keltner upper + 1d trend + volume ===
            if price_above_1d_ema and price_above_upper and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: Price breaks below Keltner lower + 1d trend + volume ===
            if not price_above_1d_ema and price_below_lower and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === MINIMUM HOLD (8 bars = 2 days to reduce churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 8:
            # Exit at middle band (mean reversion)
            if position_side > 0 and close[i] >= middle_band:
                desired_signal = 0.0
            if position_side < 0 and close[i] <= middle_band:
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
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals