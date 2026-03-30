#!/usr/bin/env python3
"""
Experiment #005: 12h RSI Extreme + 1d EMA Trend + Volume

HYPOTHESIS: RSI(14) at extreme levels (<40 for long, >60 for short) captures
cyclical mean-reversion within multi-day trends. Combined with 1d EMA trend
alignment and volume confirmation, this catches corrections that fully reverse
while avoiding false signals in trending markets.

WHY 12h: Slow enough for meaningful trends (4x daily), fast enough for
12-37 trades/year per symbol. RSI extremes are rare enough at 12h.

WHY IT WORKS IN BULL AND BEAR:
- BULL: Corrections push RSI below 40 → buy the correction in uptrend
- BEAR: Bear rallies push RSI above 60 → short the rally in downtrend
- RANGE: RSI extremes work as mean-reversion signals at boundaries

Entry: RSI extreme + volume spike (>1.5x) + trend alignment.
Stoploss: 2.0 ATR trailing.
Min hold: 3 bars (1.5 days) to avoid fee churn.

TARGET: 75-125 total trades over 4 years = 19-31/year.
Signal size: 0.30 (discrete levels to minimize fee churn).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_rsi_extreme_vol_ema50_1d_v1"
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
    
    # RSI(14)
    delta = np.zeros(n, dtype=np.float64)
    delta[1:] = close[1:] - close[:-1]
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    rsi_14 = 100.0 - (100.0 / (1.0 + rs))
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    SIZE_HALF = 0.15
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 100  # Buffer for EMA alignment
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d EMA50) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === MOMENTUM & VOLUME CONDITIONS ===
        # RSI extreme (long when oversold in uptrend, short when overbought in downtrend)
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: RSI oversold + volume + trend alignment ===
            if price_above_1d_ema and rsi_oversold and vol_spike:
                desired_signal = SIZE
            
            # === SHORT: RSI overbought + volume + trend alignment ===
            if not price_above_1d_ema and rsi_overbought and vol_spike:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
                in_position = False
                position_side = 0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
                in_position = False
                position_side = 0
        
        # === MINIMUM HOLD PERIOD (3 bars = 1.5 days to reduce churn) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 3:
            # Take profit when RSI reverts to neutral (45-55)
            if position_side > 0 and 45 < rsi_14[i] < 55:
                desired_signal = 0.0
                in_position = False
                position_side = 0
            
            if position_side < 0 and 45 < rsi_14[i] < 55:
                desired_signal = 0.0
                in_position = False
                position_side = 0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or direction flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
        
        signals[i] = desired_signal
    
    return signals