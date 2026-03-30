#!/usr/bin/env python3
"""
Experiment #022: 4h TRIX + Choppiness Regime + Volume Confirmation

HYPOTHESIS: TRIX momentum oscillator catches trend changes but generates
too many crossovers in ranging markets. Adding Choppiness Index as regime
filter (only trade when CHOP < 50) should reduce whipsaws significantly.

KEY INSIGHT:
- mtf_4h_trix_1d_ema_vol_v1 had 243 trades, Sharpe 0.439 (too many trades)
- Adding choppiness filter should reduce to ~120-180 trades
- 1d EMA as trend filter (proven to work)

WHY BOTH MARKETS:
- 2021 bull: CHOP < 50 during trends, TRIX crosses up catch rallies
- 2022 bear: CHOP < 50 during breakdowns, TRIX crosses down catch shorts
- 2025 range: CHOP > 55 = flat, avoid whipsaw crossovers

TARGET: 100-200 total trades over 4 years.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_chop_vol_v1"
timeframe = "4h"
leverage = 1.0

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

def calculate_trix(close, period=15):
    """TRIX: Triple EMA Rate of Change - momentum indicator"""
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    trix = 100 * ema3.pct_change(period)
    return trix.values

def calculate_choppiness(close, high, low, period=14):
    """Choppiness Index - distinguishes trending vs ranging markets"""
    n = len(close)
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    sum_tr = pd.Series(tr).rolling(window=period, min_periods=period).sum()
    highest = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    chop = 100 * np.log10(sum_tr / (highest - lowest)) / np.log10(period)
    return chop.fillna(50).values

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    trix_15 = calculate_trix(close, period=15)
    trix_signal = pd.Series(trix_15).ewm(span=9, min_periods=9, adjust=False).mean().values
    chop = calculate_choppiness(close, high, low, period=14)
    
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
    
    warmup = 50  # 15 for TRIX + buffer
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(trix_15[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === HTF TREND (1d EMA aligned) ===
        htf_bullish = close[i] > ema_1d_aligned[i]
        htf_bearish = close[i] < ema_1d_aligned[i]
        
        # === VOLUME SPIKE (>1.5x average) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CHOPPINESS REGIME FILTER ===
        # CHOP < 50 = trending, take signals. CHOP > 55 = choppy, skip.
        chop_trending = chop[i] < 50
        
        # === TRIX CROSSOVER SIGNALS ===
        trix_cross_up = False
        trix_cross_dn = False
        
        if i > 0 and not np.isnan(trix_signal[i-1]) and not np.isnan(trix_signal[i]):
            # TRIX crosses above signal = bullish momentum
            if trix_15[i] > trix_signal[i] and trix_15[i-1] <= trix_signal[i-1]:
                trix_cross_up = True
            # TRIX crosses below signal = bearish momentum
            if trix_15[i] < trix_signal[i] and trix_15[i-1] >= trix_signal[i-1]:
                trix_cross_dn = True
        
        # === MINIMUM HOLD: 4 bars (16h) to avoid immediate reversals ===
        min_hold_bars = 4
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === ATR TRAILING STOP (2.5x ATR from entry high/low) ===
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
            
            # Exit on trend reversal (trend filter flips)
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
            # LONG: TRIX cross up + volume spike + HTF bullish + trending market
            if trix_cross_up and vol_spike and htf_bullish and chop_trending:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: TRIX cross down + volume spike + HTF bearish + trending market
            elif trix_cross_dn and vol_spike and htf_bearish and chop_trending:
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