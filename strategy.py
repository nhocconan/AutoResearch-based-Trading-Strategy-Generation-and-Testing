#!/usr/bin/env python3
"""
Experiment #024: 12h TRIX Momentum + Choppiness Regime + 1d EMA Filter

HYPOTHESIS: 12h timeframe captures medium-term momentum swings (1-5 days)
while avoiding the fee drag of lower TFs. TRIX(12) gives smooth, lag-reduced
momentum signals. Choppiness Index filters out ranging markets where momentum
systems fail. 1d EMA adds structural trend context.

WHY 12h (not 4h): 4h strategies uniformly failed due to overtrading or neg Sharpe.
12h has proven 54% keep rate in prior experiments. Slower = fewer false signals.

WHY IT WORKS IN BOTH MARKETS:
- Bull: TRIX crosses up + price > 1d EMA = trend following
- Bear: TRIX crosses down + price < 1d EMA = shorting the trend
- Choppiness filter: avoids 2022 bottom whipsaw that destroyed prior strategies

TARGET: 75-150 total trades over 4 years (~20-40/year on 12h).
Signal size: 0.30 (discrete).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trix_chop_ema_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_trix(close, period=12):
    """TRIX - triple smoothed rate of change. Lag-reduced momentum."""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA smoothing
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, min_periods=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Rate of change of triple EMA
    trix = np.zeros(n)
    trix[0] = 0.0
    for i in range(1, n):
        if ema3.iloc[i-1] != 0:
            trix[i] = ((ema3.iloc[i] / ema3.iloc[i-1]) - 1) * 100
        else:
            trix[i] = 0.0
    
    return trix

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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values > 61.8 = choppy/range, < 38.2 = trending"""
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            if j > 0:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]))
            else:
                tr = high[j] - low[j]
            atr_sum += tr
        
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0:
            chop[i] = 100 * (np.log10(atr_sum / range_sum) / np.log10(period))
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # 1d EMA50 for trend
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1w EMA21 for longer-term trend
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    trix_12 = calculate_trix(close, period=12)
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    # TRIX signal line (EMA of TRIX)
    trix_signal = pd.Series(trix_12).ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    trailing_stop = 0.0
    
    warmup = 150  # Need enough for TRIX to stabilize
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix_12[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND CONTEXT ===
        # Bull trend: price > 1d EMA > 1w EMA
        bull_trend = close[i] > ema_1d_aligned[i] and ema_1d_aligned[i] > ema_1w_aligned[i]
        # Bear trend: price < 1d EMA < 1w EMA
        bear_trend = close[i] < ema_1d_aligned[i] and ema_1d_aligned[i] < ema_1w_aligned[i]
        trend_strong = bull_trend or bear_trend
        
        # === CHOPPINESS REGIME ===
        chop = chop_14[i]
        in_chop = chop > 61.8 if not np.isnan(chop) else False
        in_trend_regime = chop < 38.2 if not np.isnan(chop) else False
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === TRIX MOMENTUM SIGNALS ===
        trix_above_signal = trix_12[i] > trix_signal[i]
        trix_below_signal = trix_12[i] < trix_signal[i]
        
        # TRIX zero line crossover (momentum shift)
        trix_positive = trix_12[i] > 0
        trix_negative = trix_12[i] < 0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        bars_held = i - entry_bar
        
        if not in_position:
            # === LONG ENTRY ===
            # Bull trend + TRIX crosses above signal line + volume
            if bull_trend and trix_above_signal and vol_spike:
                desired_signal = SIZE
            
            # === SHORT ENTRY ===
            # Bear trend + TRIX crosses below signal line + volume
            if bear_trend and trix_below_signal and vol_spike:
                desired_signal = -SIZE
        
        # === TRAILING STOPLOSS (2.5 ATR) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            if low[i] < trailing_stop:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            if high[i] > trailing_stop:
                desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        if in_position and bars_held < 4:
            if position_side > 0:
                desired_signal = SIZE
            elif position_side < 0:
                desired_signal = -SIZE
        
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
                
                if position_side > 0:
                    trailing_stop = entry_price - 2.5 * entry_atr
                else:
                    trailing_stop = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                trailing_stop = 0.0
        
        signals[i] = desired_signal
    
    return signals