#!/usr/bin/env python3
"""
Experiment #006: 1d ATR Regime + Donchian Breakout + 1w EMA Trend

HYPOTHESIS: ATR percentile identifies high-vol trending vs low-vol choppy regimes.
In high-vol regime: trade Donchian breakouts with 1w trend filter.
In low-vol regime: stay out (too many false breakouts).
ATR-based stoploss adapts to volatility environment.

WHY 1d: Slowest practical TF = fewest trades = minimal fee drag.
WHY BOTH MARKETS: High-vol regime captures 2022 crash (vol spike = short signal)
and 2023-24 recovery (vol expansion = trend follow).

TARGET: 40-100 trades over 4 years = 10-25/year. HARD MAX: 150.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_atr_regime_donchian_1w_v1"
timeframe = "1d"
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

def atr_percentile(atr_series, period=20):
    """ATR percentile over rolling window - higher = more volatile = trending"""
    return pd.Series(atr_series).rolling(window=period, min_periods=period).apply(
        lambda x: (x[-1] - x.min()) / (x.max() - x.min() + 1e-10) if x.max() > x.min() else 0.5,
        raw=True
    ).values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 for trend direction (balance between responsiveness and smoothness)
    ema_1w_vals = df_1w['close'].values
    ema_1w = pd.Series(ema_1w_vals).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # ATR percentile (volatility regime)
    atr_pct = atr_percentile(atr_14, period=20)
    
    # Donchian channel (20 bars for 1d)
    donchian_period = 20
    rolling_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    rolling_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Volume confirmation
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = max(100, donchian_period)  # Need enough for alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if ATR percentile not ready
        if np.isnan(atr_pct[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if 1w EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === REGIME FILTER (ATR percentile) ===
        # High vol regime (pct > 0.6) = trending = enter on breakout
        # Low vol regime (pct <= 0.4) = choppy = stay out
        in_volatile_regime = atr_pct[i] > 0.5
        
        # === TREND DIRECTION (1w EMA50) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN CHANNEL ===
        channel_high = rolling_high[i]
        channel_low = rolling_low[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Price breaks above 20d high in volatile regime with trend alignment ===
            if in_volatile_regime and price_above_1w_ema and vol_spike:
                if high[i] > channel_high:
                    desired_signal = SIZE
            
            # === SHORT: Price breaks below 20d low in volatile regime with trend alignment ===
            if in_volatile_regime and not price_above_1w_ema and vol_spike:
                if low[i] < channel_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === HOLD PERIOD (minimum 3 bars to avoid churn) ===
        bars_held = i - entry_bar
        
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals