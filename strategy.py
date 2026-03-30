#!/usr/bin/env python3
"""
Experiment #011: 6h Keltner Channel + Volatility Regime + 1d Trend

HYPOTHESIS: Markets alternate between low-volatility (range-bound) and 
high-volatility (trending) regimes. Keltner channels + ATR ratio regime 
detection captures both:
- LOW vol regime: Price at outer Keltner bands often reverts
- HIGH vol regime: Keltner breakouts capture momentum moves

WHY 6h: Balances trade frequency (target 75-150 over 4 years)
Keltner is momentum-based (different from Donchian/Camarilla price-based).

1d EMA50 determines bull/bear bias. ATR(14) for stops.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 250.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_volregime_ema50_1d_v1"
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

def calculate_keltner(high, low, close, ema_period=20, atr_period=10, multiplier=2.0):
    """Keltner Channels: EMA +/- multiplier * ATR"""
    n = len(close)
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, atr_period)
    
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return ema, upper, lower

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
    
    # === Local indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    # Keltner channels (20 EMA, 10 ATR, 2x multiplier)
    keltner_mid, keltner_upper, keltner_lower = calculate_keltner(high, low, close, 20, 10, 2.0)
    
    # Volatility regime: ATR(6)/ATR(30) — need to compute 6-period ATR first
    atr_6 = calculate_atr(high, low, close, period=6)
    vol_ratio = atr_6 / np.where(atr_30 > 0, atr_30, 1)
    
    # Smooth volatility ratio with EMA for stability
    vol_ratio_smooth = pd.Series(vol_ratio).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Bollinger bandwidth for regime confirmation
    bb_mid = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / np.where(bb_mid > 0, bb_mid, 1)
    
    # BB width percentile (20-day)
    bb_width_pct = np.zeros(n)
    for i in range(20, n):
        window = bb_width[max(0, i-20):i+1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            bb_width_pct[i] = (valid < bb_width[i]).sum() / len(valid)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough for all indicators
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(vol_ratio_smooth[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        bull_trend = close[i] > ema_1d_aligned[i]
        bear_trend = close[i] < ema_1d_aligned[i]
        
        # === VOLATILITY REGIME ===
        # vol_ratio_smooth > 0.9: trending/volatile, < 0.7: calm/range
        high_vol = vol_ratio_smooth[i] > 0.9
        low_vol = vol_ratio_smooth[i] < 0.7
        
        # BB squeeze detection (low volatility confirmed)
        bb_squeeze = bb_width_pct[i] < 0.25
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # Price position relative to Keltner
        price_above_upper = close[i] > keltner_upper[i]
        price_below_lower = close[i] < keltner_lower[i]
        price_above_mid = close[i] > keltner_mid[i]
        price_below_mid = close[i] < keltner_mid[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY CONDITIONS ===
            if bull_trend:
                # Option 1: Breakout in high vol (aggressive)
                if high_vol and price_above_upper and vol_spike:
                    desired_signal = SIZE
                # Option 2: Mean reversion from BB squeeze in low vol
                elif low_vol and bb_squeeze and price_below_lower and vol_spike:
                    desired_signal = SIZE
                # Option 3: Pullback to mid Keltner in uptrend
                elif price_above_mid and price_below_mid == False and vol_spike:
                    # Wait, this doesn't make sense. Let me fix:
                    pass
            
            # === SHORT ENTRY CONDITIONS ===
            if bear_trend:
                # Option 1: Breakout in high vol (aggressive)
                if high_vol and price_below_lower and vol_spike:
                    desired_signal = -SIZE
                # Option 2: Mean reversion from BB squeeze in low vol
                elif low_vol and bb_squeeze and price_above_upper and vol_spike:
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
        
        # === EXIT ON REGIME CHANGE (volatility crush) ===
        if in_position and position_side > 0:
            # Exit if volatility contracts after entry (take profit)
            if vol_ratio_smooth[i] < 0.7 and bb_width_pct[i] < 0.3:
                if i - entry_bar >= 4:  # Hold at least 1 day
                    desired_signal = 0.0
        
        if in_position and position_side < 0:
            if vol_ratio_smooth[i] < 0.7 and bb_width_pct[i] < 0.3:
                if i - entry_bar >= 4:
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