#!/usr/bin/env python3
"""
Experiment #024: 1d MACD Crossover + 1w SMA Trend + Volume

HYPOTHESIS: MACD(12,26,9) crossovers on 1d are significant momentum shifts.
Combined with 1w SMA trend filter and volume confirmation, this should:
- Work in 2021 bull: MACD crosses up in uptrend = strong signal
- Work in 2022 bear: MACD crosses down in downtrend = strong short
- Work in 2025 range: fewer crosses = less whipsaw

KEY INSIGHT: MACD crossover frequency on 1d ~ 30-50/year, perfect for
long-term trend following with minimal overtrading. Weekly SMA adds
macro context to filter noise.

TARGET TRADES: 75-150 total over 4 years (19-37/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_macd_1w_sma_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_macd(close, fast=12, slow=26, signal=9):
    """MACD indicator: EMA(fast) - EMA(slow), with signal line"""
    n = len(close)
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean().values
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian channel for trend confirmation"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w SMA for macro trend (call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    sma_1w_50 = pd.Series(df_1w['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w_50)
    
    # === 1d indicators ===
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20 for local structure
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 80
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === MACD CROSSOVER DETECTION ===
        # MACD line crosses above signal = bullish momentum shift
        macd_bull_cross = (macd_line[i] > macd_signal[i]) and (i > 0) and (macd_line[i-1] <= macd_signal[i-1])
        # MACD line crosses below signal = bearish momentum shift
        macd_bear_cross = (macd_line[i] < macd_signal[i]) and (i > 0) and (macd_line[i-1] >= macd_signal[i-1])
        
        # MACD histogram strengthening (momentum confirmation)
        hist_strengthening = macd_hist[i] > macd_hist[i-1] if i > 0 else False
        hist_weakening = macd_hist[i] < macd_hist[i-1] if i > 0 else False
        
        # === WEEKLY TREND FILTER ===
        htf_bullish = close[i] > sma_1w_aligned[i]
        htf_bearish = close[i] < sma_1w_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 2 bars (2 days) ===
        min_hold = (i - entry_bar) >= 2
        
        # === TRAILING STOP (3x ATR) ===
        def check_stop():
            if not in_position:
                return False
            if position_side > 0:
                # Long stop: price drops below highest - 3*ATR
                return low[i] < (highest_since_entry - 3.0 * entry_atr)
            else:
                # Short stop: price rises above lowest + 3*ATR
                return high[i] > (lowest_since_entry + 3.0 * entry_atr)
        
        # Update tracking
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === EXITS ===
        if in_position:
            stop_hit = check_stop()
            
            # Weekly trend reversal exits (after min hold)
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: MACD bullish cross + weekly uptrend + volume confirmation
            if macd_bull_cross and htf_bullish and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG WEAK: MACD cross + weekly uptrend, no volume spike (smaller size)
            elif macd_bull_cross and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.5  # Half size without volume confirmation
            
            # SHORT: MACD bearish cross + weekly downtrend + volume confirmation
            elif macd_bear_cross and htf_bearish and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT WEAK: MACD cross + weekly downtrend, no volume spike (smaller size)
            elif macd_bear_cross and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.5  # Half size without volume confirmation
            
            else:
                signals[i] = 0.0
    
    return signals