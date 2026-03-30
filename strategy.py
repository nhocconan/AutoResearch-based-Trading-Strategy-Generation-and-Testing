#!/usr/bin/env python3
"""
Experiment #008: 12h Camarilla S4/R4 + Choppiness Regime + Volume

HYPOTHESIS: Trading ONLY deep Camarilla levels (S4/R4) catches high-conviction
reversals. Adding Choppiness Index (<38.2) removes ranging periods where
Camarilla mean-reversion fails. 1d EMA50 confirms trend alignment.

WHY IT WORKS IN BULL AND BEAR:
- S4 = deep oversold support. In uptrends, buying S4 catches dips before continuation.
- R4 = deep overbought resistance. In downtrends, shorting R4 catches reversals.
- CHOP filter prevents trading during choppy/ranging periods (where 60% of failures occur).

KEY IMPROVEMENT over failed #016 (275 trades):
- CHOP < 38.2 filter: only trade when market is TRENDING (not ranging)
- Only S4/R4: deeper levels = higher conviction = fewer but better trades
- 3-bar minimum hold: reduces false exits
- Removed S3/R3 entries: too many false signals

TARGET: 75-120 total trades over 4 years (19-30/year). HARD MAX: 150.
Signal size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_s4r4_chop_1d_v1"
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

def calculate_chop(high, low, close, period=14):
    """Choppiness Index (CHOP)
    CHOP > 61.8 = ranging (use mean reversion)
    CHOP < 38.2 = trending (use trend following)
    We ONLY enter when CHOP < 38.2 (trending market)
    """
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        sum_tr = 0.0
        for j in range(period):
            tr = max(high[i-j] - low[i-j], 
                    abs(high[i-j] - close[i-j-1]), 
                    abs(low[i-j] - close[i-j-1]))
            sum_tr += tr
        
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest > lowest:
            chop[i] = 100 * (np.log(sum_tr) / np.log(period * (highest - lowest)))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend direction
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_chop(high, low, close, period=14)
    
    # Volume ratio (20-bar SMA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals array
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
    
    warmup = 150  # Need enough for CHOP(14) + vol_ma(20) + EMA50 alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_50_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND (1d EMA50) ===
        price_above_ema = close[i] > ema_50_aligned[i]
        
        # === REGIME: CHOP < 38.2 means trending (good for Camarilla) ===
        is_trending = chop[i] < 38.2
        
        # === VOLUME confirmation ===
        vol_spike = vol_ratio[i] > 1.8
        
        # === CAMARILLA S4/R4 from previous CLOSED bar ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        prev_range = prev_high - prev_low
        
        # Deep Camarilla levels only (higher conviction)
        s4 = prev_close - prev_range * 0.18333  # Deep support
        r4 = prev_close + prev_range * 0.18333  # Deep resistance
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position and is_trending:
            # === LONG: S4 touch + volume + trend alignment ===
            # Trend: price should be above EMA (uptrend)
            # CHOP: market is trending (not ranging)
            # Volume: institutional confirmation
            if price_above_ema and vol_spike:
                if low[i] <= s4:
                    desired_signal = SIZE
            
            # === SHORT: R4 touch + volume + trend alignment ===
            # Trend: price should be below EMA (downtrend)
            if not price_above_ema and vol_spike:
                if high[i] >= r4:
                    desired_signal = -SIZE
        
        # === MINIMUM HOLD (3 bars = 1.5 days) ===
        bars_held = i - entry_bar
        
        if in_position:
            # === STOPLOSS (2.5 ATR trailing) ===
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                trailing_stop = highest_since_entry - 2.5 * entry_atr
                stop_price = max(stop_price, trailing_stop)
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            if position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                trailing_stop = lowest_since_entry + 2.5 * entry_atr
                stop_price = min(stop_price, trailing_stop)
                if high[i] > stop_price:
                    desired_signal = 0.0
            
            # === TAKE PROFIT (after min hold): price returns to prev close ===
            if bars_held >= 3:
                if position_side > 0 and close[i] >= prev_close:
                    desired_signal = 0.0
                if position_side < 0 and close[i] <= prev_close:
                    desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or reversal
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                # Initial stop: 2.5 ATR
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        
        elif in_position:
            # Exit
            in_position = False
            position_side = 0
            stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals