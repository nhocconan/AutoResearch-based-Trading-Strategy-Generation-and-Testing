#!/usr/bin/env python3
"""
Experiment #023: 4h Donchian Breakout + EMA Momentum + Volume Spike

HYPOTHESIS: Donchian(20) breakouts capture momentum shifts when price breaks
the 20-bar (5-day) channel. Combining with:
- EMA(8/21) crossover for momentum confirmation
- Volume spike (1.5x) for institutional validation
- 1d EMA50 for trend direction alignment
- ATR stoploss for risk management

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull market: Breakouts above EMA uptrend capture momentum continuation
- Bear market: Breakouts below EMA downtrend capture bearish momentum
- Symmetrical: Long on upward breakouts, short on downward breakouts
- Stoploss at 2x ATR handles volatility expansion in both directions

TRADE TARGET: 75-200 total over 4 years (19-50/year). Discrete size 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_ema_momentum_vol_v1"
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
    
    # 1d EMA50 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # EMA crossover (fast/slow for momentum)
    ema_fast = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Donchian channel (20 bars = 5 days)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Pre-compute indicators
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Need enough bars for all indicators
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if HTF EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === CONFIRMATIONS ===
        # 1d trend direction
        price_above_1d = close[i] > ema_1d_aligned[i]
        
        # EMA momentum (fast crosses above slow = bullish momentum)
        ema_bullish = ema_fast[i] > ema_slow[i]
        
        # Volume spike
        vol_spike = vol_ratio[i] > 1.5
        
        # Donchian breakout (price breaks above upper or below lower band)
        donch_break_up = close[i] > donchian_upper[i-1]  # Previous bar's upper (no look-ahead)
        donch_break_dn = close[i] < donchian_lower[i-1]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Donchian breakout UP + bullish EMA + volume spike
            if donch_break_up and ema_bullish and vol_spike:
                desired_signal = SIZE
            
            # SHORT: Donchian breakout DOWN + bearish EMA + volume spike
            if donch_break_dn and not ema_bullish and vol_spike:
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
        
        # === MINIMUM HOLD (4 bars = 1 day to avoid churn) ===
        bars_held = i - entry_bar
        if bars_held < 4:
            # Don't exit due to opposite signal
            if in_position and desired_signal != 0.0 and np.sign(desired_signal) != position_side:
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
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals