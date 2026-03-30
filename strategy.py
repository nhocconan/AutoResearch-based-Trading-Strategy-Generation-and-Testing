#!/usr/bin/env python3
"""
Experiment #008: 12h TRIX Reversal + 1w Trend + Donchian Structure

HYPOTHESIS: TRIX momentum indicator generates reliable reversal signals when
it crosses zero from oversold (<0) or overbought (>0) zones. Combined with:
- 1w SMA(50) for macro trend direction (prevents fighting the weekly trend)
- Donchian(20) on 12h for local structure (validates the reversal point)
- Volume spike for confirmation
- ATR stoploss (2.5x) for risk management

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull (2021): TRIX crossing up from oversold at S3/S4 = strong bounce setup
- Bear (2022): TRIX crossing down from overbought at R3/R4 = fade rallies
- Range (2025): TRIX zero-line crossovers work well in choppy markets
- 1w filter ensures we don't short bull dips or long bear rallies

TARGET: 60-120 trades over 4 years (15-30/year). Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trix_donchian_1w_vol_v1"
timeframe = "12h"
leverage = 1.0

def calculate_trix(close, period=14):
    """
    TRIX (Triple EMA) - momentum oscillator.
    TRIX = Rate of change of triple EMA.
    Signal line = EMA of TRIX.
    """
    if len(close) < period * 3:
        return np.full(len(close), np.nan), np.full(len(close), np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple EMA = TRIX
    trix = np.zeros(len(close), dtype=np.float64)
    for i in range(1, len(close)):
        if not np.isnan(ema3.iloc[i]) and not np.isnan(ema3.iloc[i-1]) and ema3.iloc[i-1] != 0:
            trix[i] = ((ema3.iloc[i] / ema3.iloc[i-1]) - 1) * 100
        else:
            trix[i] = np.nan
    
    # Signal line = EMA of TRIX
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, min_periods=9, adjust=False).mean().values
    
    return trix, trix_signal

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
    """Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    middle = (upper + lower) / 2.0
    return upper, lower, middle

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
    
    # === 12h indicators ===
    trix, trix_signal = calculate_trix(close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian 20 for structure
    dc_upper_20, dc_lower_20, dc_mid_20 = calculate_donchian(high, low, period=20)
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # TRIX histogram for momentum direction
    trix_hist = trix - trix_signal
    
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
    
    warmup = 100
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION (1w) ===
        htf_bullish = close[i] > sma_1w_aligned[i]
        htf_bearish = close[i] < sma_1w_aligned[i]
        
        # === TRIX SIGNALS ===
        # Zero-line crossover with momentum confirmation
        trix_above_zero = trix[i] > 0
        trix_below_zero = trix[i] < 0
        
        # Momentum strengthening (histogram rising)
        trix_rising = trix_hist[i] > trix_hist[i-1] if not np.isnan(trix_hist[i-1]) else False
        trix_falling = trix_hist[i] < trix_hist[i-1] if not np.isnan(trix_hist[i-1]) else False
        
        # === DONCHIAN STRUCTURE ===
        near_dc_lower = close[i] < dc_lower_20[i] * 1.05 if not np.isnan(dc_lower_20[i]) else False
        near_dc_upper = close[i] > dc_upper_20[i] * 0.95 if not np.isnan(dc_upper_20[i]) else False
        near_dc_mid = abs(close[i] - dc_mid_20[i]) < dc_mid_20[i] * 0.02 if not np.isnan(dc_mid_20[i]) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === MINIMUM HOLD: 1 bar (12h) ===
        min_hold = (i - entry_bar) >= 1
        
        # === ATR TRAILING STOP (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Trend reversal exit (1w trend flip)
            if position_side > 0 and htf_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and htf_bullish and min_hold:
                stop_hit = True
            
            # TRIX reversal exit
            if position_side > 0 and trix_falling and trix_above_zero and min_hold:
                stop_hit = True
            if position_side < 0 and trix_rising and trix_below_zero and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: TRIX crosses above zero from below + near Donchian lower + 1w uptrend
            # This catches reversals at support
            long_trix_cross = (trix[i] > 0 and trix[i-1] <= 0) if not np.isnan(trix[i-1]) else False
            long_momentum = trix_rising or trix_hist[i] > 0
            
            if long_trix_cross and near_dc_lower and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG ALT: TRIX already positive + strong momentum + near DC lower + vol spike
            elif trix_above_zero and trix_rising and near_dc_lower and htf_bullish and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: TRIX crosses below zero from above + near Donchian upper + 1w downtrend
            short_trix_cross = (trix[i] < 0 and trix[i-1] >= 0) if not np.isnan(trix[i-1]) else False
            short_momentum = trix_falling or trix_hist[i] < 0
            
            if short_trix_cross and near_dc_upper and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT ALT: TRIX already negative + falling momentum + near DC upper + vol spike
            elif trix_below_zero and trix_falling and near_dc_upper and htf_bearish and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals