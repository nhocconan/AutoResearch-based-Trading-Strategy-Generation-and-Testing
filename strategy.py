#!/usr/bin/env python3
"""
Experiment #022: 12h RSI4 Mean-Reversion + 1d EMA Trend

HYPOTHESIS: Fast RSI(4) at extremes catches reversals in BOTH bull and bear markets.
- Bull market: buy when RSI(4) < 20 (oversold) + price above 1d EMA50
- Bear market: short when RSI(4) > 80 (overbought) + price below 1d EMA50
- Volume confirms the reversal momentum
- ATR stop protects against trend continuation

WHY IT WORKS IN BOTH MARKETS:
- Bull: buy dips to oversold = high win rate on reversals
- Bear: short rallies to overbought = catches the bounce top
- Symmetric entry logic = works regardless of direction

TARGET: 75-150 total trades over 4 years (19-37/year). HARD MAX: 200.
Signal size: 0.25. Stop: 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi4_meanrev_ema50_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_rsi(prices, period=4):
    """Fast RSI for mean-reversion signals"""
    n = len(prices)
    if n < period + 1:
        return np.full(n, np.nan)
    
    deltas = np.diff(prices, prepend=prices[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

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
    rsi_4 = calculate_rsi(close, period=4)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Buffer for EMA50 alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(rsi_4[i]) or np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            # === LONG: RSI(4) oversold + above 1d EMA + volume ===
            if price_above_1d_ema and rsi_4[i] < 20 and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: RSI(4) overbought + below 1d EMA + volume ===
            if not price_above_1d_ema and rsi_4[i] > 80 and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
            
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD (2 bars = 1 day to reduce churn) ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 2:
            # Exit if RSI reverts to neutral (40-60)
            if position_side > 0 and 40 < rsi_4[i] < 60:
                desired_signal = 0.0
                in_position = False
                position_side = 0
            if position_side < 0 and 40 < rsi_4[i] < 60:
                desired_signal = 0.0
                in_position = False
                position_side = 0
        
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