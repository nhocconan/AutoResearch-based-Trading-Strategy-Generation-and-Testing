#!/usr/bin/env python3
"""
Experiment #010: 1d Supertrend Breakout + Weekly EMA + Volume

HYPOTHESIS: Supertrend(ATR=10, mult=3) is a volatility-adaptive trend indicator
that hasn't been tested in this session. It provides:
- Dynamic support/resistance that adjusts for volatility
- Symmetrical signals for long and short
- Built-in trailing stop (no manual stoploss needed)

Weekly EMA200 confirms structural trend direction, filtering whipsaws.
Volume spike confirms institutional participation.

WHY IT WORKS IN BULL AND BEAR:
- Supertrend goes long when price crosses above the upper band
- Supertrend goes short when price crosses below the lower band
- Symmetrical design works in both directions
- Weekly filter prevents buying breakdowns in downtrends

TARGET: 75-150 total trades over 4 years. SIZE: 0.25.
Signal requires: Supertrend direction + Weekly trend alignment + Volume spike.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_supertrend_vol_weekly_ema_v1"
timeframe = "1d"
leverage = 1.0

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Supertrend indicator calculation.
    Returns: supertrend values (positive = bullish, negative = bearish)
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, 0.0)
    
    # Calculate ATR
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate Supertrend
    supertrend = np.zeros(n, dtype=np.float64)
    trend = np.ones(n, dtype=np.int32)  # 1 = uptrend, -1 = downtrend
    final_upper = np.zeros(n, dtype=np.float64)
    final_lower = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if i < 1:
            continue
        
        hl2 = (high[i] + low[i]) / 2.0
        upper = hl2 + multiplier * atr[i]
        lower = hl2 - multiplier * atr[i]
        
        # Upper band
        if close[i-1] > final_upper[i-1]:
            final_upper[i] = max(upper, final_upper[i-1])
        else:
            final_upper[i] = upper
        
        # Lower band
        if close[i-1] < final_lower[i-1]:
            final_lower[i] = min(lower, final_lower[i-1])
        else:
            final_lower[i] = lower
        
        # Trend direction
        if close[i] > final_upper[i-1]:
            trend[i] = 1
        elif close[i] < final_lower[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]
        
        # Supertrend value: positive in uptrend, negative in downtrend
        if trend[i] == 1:
            supertrend[i] = final_lower[i]
        else:
            supertrend[i] = -final_upper[i]
    
    return supertrend

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load Weekly data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA200 for structural trend (requires ~1000 bars, enough for 1d)
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Pre-calculate ALL 1d indicators before loop ===
    # Supertrend (period=10, multiplier=3)
    supertrend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ATR for stoploss
    atr_14 = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    prev_supertrend = 0.0
    
    warmup = 250  # Need enough for Supertrend(10) + EMA200 alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if Weekly EMA not aligned
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        current_supertrend = supertrend[i]
        
        # === WEEKLY TREND FILTER ===
        # Only long when price above weekly EMA (uptrend)
        # Only short when price below weekly EMA (downtrend)
        price_above_weekly = close[i] > ema_1w_aligned[i]
        price_below_weekly = close[i] < ema_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === SUPERTREND CROSSOVER DETECTION ===
        # Long signal: Supertrend crosses from negative to positive
        # Short signal: Supertrend crosses from positive to negative
        bullish_cross = (prev_supertrend < 0) and (current_supertrend > 0)
        bearish_cross = (prev_supertrend > 0) and (current_supertrend < 0)
        
        desired_signal = 0.0
        
        if not in_position:
            # === ENTRY LOGIC ===
            # Long: Supertrend crosses bullish + above weekly EMA + volume spike
            if bullish_cross and price_above_weekly and vol_spike:
                desired_signal = SIZE
            
            # Short: Supertrend crosses bearish + below weekly EMA + volume spike
            if bearish_cross and price_below_weekly and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR from entry) ===
        if in_position and position_side > 0:
            stop_price = entry_price - 2.0 * entry_atr
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            stop_price = entry_price + 2.0 * entry_atr
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (3:1 R/R) ===
        if in_position:
            bars_held = i - entry_bar
            if bars_held >= 3:  # Hold at least 3 bars (3 days)
                if position_side > 0:
                    profit_target = entry_price + 3.0 * entry_atr
                    if high[i] >= profit_target:
                        desired_signal = 0.0
                if position_side < 0:
                    profit_target = entry_price - 3.0 * entry_atr
                    if low[i] <= profit_target:
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
        
        prev_supertrend = current_supertrend
        signals[i] = desired_signal
    
    return signals