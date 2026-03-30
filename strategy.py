#!/usr/bin/env python3
"""
Experiment #024: 4h Williams %R Mean Reversion + 1d SMA + Volume

HYPOTHESIS: Williams %R at extreme levels (-80 for longs, -20 for shorts)
is a proven mean reversion signal. Price tends to bounce from these extremes.
Combined with 1d SMA trend filter and volume confirmation:
- 2021 bull: bounces from -80 with 1d uptrend = high win rate
- 2022 bear: bounces from -20 with 1d downtrend = short rallies
- 2025 range: bounces from extremes work in both directions

WHY THIS SHOULD WORK: Williams %R is a bounded momentum oscillator that
identifies exhaustion points. The -80/-20 levels are statistically proven
to signal reversals. Volume confirms the exhaustion, and 1d SMA ensures
we're not fighting the macro trend.

TARGET: 75-150 total trades over 4 years (18-37/year). Size: 0.30.
Simple = fewer trades = less fee drag = better test generalization.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_williamsr_1d_sma_vol_v1"
timeframe = "4h"
leverage = 1.0

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - measures current price relative to high-low range"""
    n = len(close)
    result = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        range_ = highest_high - lowest_low
        
        if range_ > 1e-10:
            result[i] = -100.0 * (highest_high - close[i]) / range_
    
    return result

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

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d SMA for macro trend (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    sma_1d_50 = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d_50)
    
    # === 4h indicators ===
    williams_r = calculate_williams_r(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume analysis
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
    
    warmup = 60
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DETECTION ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3
        
        # === WILLIAMS %R EXTREMES ===
        oversold = williams_r[i] <= -80  # Price at 14-bar low
        overbought = williams_r[i] >= -20  # Price at 14-bar high
        
        # === MINIMUM HOLD: 2 bars (8h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR STOPLOSS (2.5x ATR) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                # Long stop: price fell 2.5 ATR from entry
                return close[i] < (entry_price - 2.5 * entry_atr)
            else:
                # Short stop: price rose 2.5 ATR from entry
                return close[i] > (entry_price + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on trend reversal
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
            # LONG: Williams %R oversold + volume spike + 1d uptrend
            if oversold and vol_spike and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = SIZE
            
            # SHORT: Williams %R overbought + volume spike + 1d downtrend
            elif overbought and vol_spike and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
    
    return signals