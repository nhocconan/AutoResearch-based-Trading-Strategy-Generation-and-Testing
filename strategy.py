#!/usr/bin/env python3
"""
Experiment #025: 12h Weekly Trend Aligned Donchian Breakout

HYPOTHESIS: Single condition entry + HTF trend filter + volume confirmation.
The previous TRIX+Camarilla+Donchian combo was too complex (172 trades, -0.215 Sharpe).
This strips to ONE breakout signal, weekly trend filter, and volume confirmation.

WHY 12h + 1w:
- 12h: 50-150 target trades over 4 years (12-37/year), HARD MAX 200
- 1w: Weekly trend filter for structural direction (not daily noise)
- Donchian(20) on 12h = 10-day breakout window, proven in DB

WHY IT WORKS IN BULL + BEAR:
- Bull: 1w EMA50 up + 12h Donchian breakout + vol = ride the trend
- Bear: 1w EMA50 down + 12h Donchian breakdown + vol = short rallies
- Range: No trades when 1w EMA flat or price inside Donchian (chop)

TARGET: 75-120 total trades over 4 years.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1w_ema_vol_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - shift(1) for completed bars"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    middle = (upper + lower) / 2
    return upper, middle, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA50 for structural trend (align to 12h bars)
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_up, donchian_mid, donchian_lo = calculate_donchian(high, low, period=20)
    
    # Volume ratio (20-period MA)
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
    
    warmup = max(200, 20 * 28)  # Donchian(20) + EMA50 on weekly aligned
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === WEEKLY TREND FILTER ===
        ema_50_w = ema_50_aligned[i]
        weekly_bullish = close[i] > ema_50_w
        weekly_bearish = close[i] < ema_50_w
        
        # === DONCHIAN BREAKOUT (use pre-shifted values) ===
        dc_up = donchian_up[i]
        dc_mid = donchian_mid[i]
        dc_lo = donchian_lo[i]
        
        # Skip if Donchian not ready
        if np.isnan(dc_up) or np.isnan(dc_lo):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # Breakout detection
        donchian_break_up = close[i] > dc_up
        donchian_break_down = close[i] < dc_lo
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Weekly trend up + 12h breakout + volume ===
            if weekly_bullish and donchian_break_up and vol_spike:
                desired_signal = SIZE
            
            # === SHORT ENTRY: Weekly trend down + 12h breakdown + volume ===
            if weekly_bearish and donchian_break_down and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR from entry) ===
        if in_position:
            if position_side > 0:
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                # Take profit: price reaches Donchian upper + some profit
                elif close[i] > dc_up and close[i] > entry_price + 2 * entry_atr:
                    desired_signal = SIZE / 2  # Take half profit
            
            elif position_side < 0:
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                # Take profit: price reaches Donchian lower + some profit
                elif close[i] < dc_lo and close[i] < entry_price - 2 * entry_atr:
                    desired_signal = -SIZE / 2  # Take half profit
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
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