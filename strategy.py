#!/usr/bin/env python3
"""
Experiment #021: 4h Donchian Breakout + Volume + 1d EMA21 Trend

HYPOTHESIS: Classic Donchian(20) breakout captures momentum explosions.
By requiring:
1. Close BREAKS ABOVE/BELOW Donchian(20) (not just touches)
2. Volume confirmation (ratio > 1.5x)
3. 1d EMA21 trend alignment
4. ATR regime filter (ATR percentile)

This catches high-probability momentum continuations while avoiding whipsaws.

WHY IT WORKS IN BULL AND BEAR: Symmetrical breakout logic — long breakouts
in uptrends (price > 1d EMA), short breakouts in downtrends. The 1d trend
filter keeps you on the right side of the major trend.

TARGET: 75-150 total trades over 4 years (19-37/year).
Signal size: 0.30.

LEARNED FROM FAILURES: 
- 12h Camarilla strategies failed (Sharpe -0.6 to -1.6)
- Too-tight entries = 0 trades
- Volume confirmation is essential
- 4h is the sweet spot (proven in DB: Sharpe 1.1-1.5)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_ema21_1d_v2"
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
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channel (20 periods = 5 days)
    donchian_period = 20
    upper_band = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_band = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    mid_band = (upper_band + lower_band) / 2.0
    
    # Volume ratio (20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR regime: compare current ATR to 50-period ATR percentile
    atr_ma = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / np.where(atr_ma > 0, atr_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for rolling calculations
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if ATR regime not ready
        if np.isnan(atr_ratio[i]) or atr_ratio[i] < 0.7:
            # Low volatility - skip, could be consolidation
            signals[i] = 0.0
            if in_position:
                # Keep existing position
                pass
            else:
                continue
        
        # === TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT CHECK ===
        # Must have CLOSE above/below band, not just touching
        close_above_upper = close[i] > upper_band[i-1] if i > 0 else False
        close_below_lower = close[i] < lower_band[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Donchian upper with volume + trend alignment ===
            if price_above_1d_ema and vol_spike and close_above_upper:
                desired_signal = SIZE
            
            # === SHORT: Breakdown below Donchian lower with volume + trend alignment ===
            if not price_above_1d_ema and vol_spike and close_below_lower:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR from entry) ===
        if in_position and position_side > 0:
            # Trailing stop: highest high since entry - 2.5 * ATR
            if high[i] > prices['high'].iloc[entry_bar:i+1].max():
                # Update stop only if new high
                new_stop = high[i] - 2.5 * entry_atr
                current_stop = entry_price - 2.5 * entry_atr  # Initial stop
                # Trail stop up but never down
                trailing_stop = max(current_stop, new_stop)
            else:
                trailing_stop = entry_price - 2.5 * entry_atr
            
            # Check stoploss
            if low[i] < (entry_price - 2.5 * entry_atr):
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            # Trailing stop: lowest low since entry + 2.5 * ATR
            if low[i] < prices['low'].iloc[entry_bar:i+1].min():
                new_stop = low[i] + 2.5 * entry_atr
                current_stop = entry_price + 2.5 * entry_atr
                trailing_stop = min(current_stop, new_stop)
            else:
                trailing_stop = entry_price + 2.5 * entry_atr
            
            # Check stoploss
            if high[i] > (entry_price + 2.5 * entry_atr):
                desired_signal = 0.0
        
        # === HOLD MINIMUM 2 BARS to avoid churn ===
        bars_held = i - entry_bar
        if in_position and bars_held < 2:
            # Force hold for at least 2 bars
            if position_side > 0 and desired_signal == 0.0:
                # Don't exit early
                desired_signal = SIZE
            if position_side < 0 and desired_signal == 0.0:
                desired_signal = -SIZE
        
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