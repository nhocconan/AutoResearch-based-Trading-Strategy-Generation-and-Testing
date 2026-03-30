#!/usr/bin/env python3
"""
Experiment #006: 12h TRIX Momentum + Volume + 1d EMA Trend

HYPOTHESIS: TRIX is a triple-smoothed momentum oscillator that catches trend changes
early while filtering noise. Combining TRIX crossover with volume confirmation and
1d EMA50 trend alignment identifies high-probability momentum shifts.

WHY 12h: 3x slower than 4h = fewer but higher quality signals.
TRIX triple smoothing reduces noise vs single/double EMA.
Crossover signals are discrete = less churn than continuous oscillators.

WHY IT WORKS IN BULL AND BEAR: TRIX crossover captures both the start of rallies
and the beginning of selloffs. EMA50 filter ensures trading with the higher timeframe
trend, avoiding counter-trend trades during consolidations.

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trix_vol_ema50_1d_v1"
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
    
    # TRIX(9) - triple EMA momentum
    period_trix = 9
    ema1 = pd.Series(close).ewm(span=period_trix, min_periods=period_trix, adjust=False).mean()
    ema2 = ema1.ewm(span=period_trix, min_periods=period_trix, adjust=False).mean()
    ema3 = ema2.ewm(span=period_trix, min_periods=period_trix, adjust=False).mean()
    trix_val = ema3.pct_change(period=1) * 100  # rate of change of triple EMA
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    prev_trix = 0.0
    entry_bar = 0
    
    warmup = 200  # Need enough for TRIX triple EMA + alignment
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if TRIX or EMA not aligned
        if np.isnan(trix_val.iloc[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        trix = trix_val.iloc[i]
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: TRIX crosses above 0 + volume + trend alignment ===
            if prev_trix <= 0 and trix > 0 and price_above_1d_ema and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: TRIX crosses below 0 + volume + trend alignment ===
            if prev_trix >= 0 and trix < 0 and not price_above_1d_ema and vol_spike:
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
        
        # === TAKE PROFIT: TRIX reverses ===
        if in_position:
            # Exit long when TRIX turns down
            if position_side > 0 and prev_trix > 0 and trix < 0:
                desired_signal = 0.0
            # Exit short when TRIX turns up
            if position_side < 0 and prev_trix < 0 and trix > 0:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = close[i] - 2.0 * entry_atr
                else:
                    stop_price = close[i] + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        # Update TRIX for next bar
        prev_trix = trix
        signals[i] = desired_signal
    
    return signals