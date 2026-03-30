#!/usr/bin/env python3
"""
Experiment #021: 12h Donchian Breakout + Volume Spike + 1d EMA Trend

HYPOTHESIS: Donchian breakout captures momentum shifts when price breaks
multi-day ranges. By requiring:
1. Close beyond 12h Donchian bands (momentum confirmation)
2. Volume spike on breakout bar (institutional participation)
3. 1d EMA alignment (trend filter)

This catches trending moves in both directions without overtrading.

WHY IT WORKS BOTH MARKETS:
- Bull: Breakout above upper band + price > 1d EMA = strong uptrend continuation
- Bear: Breakout below lower band + price < 1d EMA = strong downtrend continuation
- Range: Breakouts fail quickly, volume spike filters false breakouts

TARGET: 75-150 total trades over 4 years. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_breakout_vol_1d_v1"
timeframe = "12h"
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
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = ~10 days on 12h)
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Volume ratio (20-bar average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = max(100, donchian_period + 20)
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Price CLOSES beyond bands (shifted by 1, so no look-ahead)
        breakout_above = close[i] > upper_band[i]
        breakout_below = close[i] < lower_band[i]
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # LONG: Breakout above upper band + volume spike + trend aligned
            if breakout_above and vol_spike and price_above_1d_ema:
                desired_signal = SIZE
            
            # SHORT: Breakout below lower band + volume spike + trend aligned
            if breakout_below and vol_spike and not price_above_1d_ema:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR) ===
        if in_position and position_side > 0:
            stop_loss = entry_price - 2.0 * entry_atr
            if low[i] < stop_loss:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            stop_loss = entry_price + 2.0 * entry_atr
            if high[i] > stop_loss:
                desired_signal = 0.0
        
        # === EXIT: Opposite Donchian band (trend exhaustion) ===
        # Only exit if we're in position and price reverses to other band
        if in_position and position_side > 0:
            # Exit long if we break below lower band (reversal signal)
            if breakout_below and vol_spike:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Exit short if we break above upper band (reversal signal)
            if breakout_above and vol_spike:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
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