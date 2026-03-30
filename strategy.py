#!/usr/bin/env python3
"""
Experiment #024: 6h Keltner Channel Breakout + 1d SMA + Volatility-Adjusted ATR Stop

HYPOTHESIS: Keltner Channel breakouts identify when price finally escapes
consolidation. Unlike Camarilla (mean reversion at pivot levels), this catches
the START of new moves. Combined with 1d SMA trend filter and volume confirmation:
- 2021 bull: Keltner breakouts above 1d SMA capture trending continuations
- 2022 bear: Breakouts below 1d SMA catch short-side moves after consolidations
- 2025 range: False breakouts filtered by requiring strong volume

KEY INSIGHT: Previous Camarilla (#007) fades extremes. This ADDS a breakout
strategy for 6h - different edge, same timeframe, complementary signals.

TRADE COUNT: 100-200 total over 4 years (25-50/year). HARD MAX: 300.
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_keltner_breakout_1d_sma_v1"
timeframe = "6h"
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

def calculate_keltner(high, low, close, ema_period=20, atr_period=20, multiplier=2.5):
    """
    Keltner Channel:
    Middle = EMA(close, period)
    Upper = EMA + multiplier * ATR
    Lower = EMA - multiplier * ATR
    """
    ema = pd.Series(close).ewm(span=ema_period, min_periods=ema_period, adjust=False).mean().values
    atr = calculate_atr(high, low, close, period=atr_period)
    
    upper = ema + multiplier * atr
    lower = ema - multiplier * atr
    
    return upper, ema, lower

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
    
    # === 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    kelt_upper, kelt_mid, kelt_lower = calculate_keltner(high, low, close, 
                                                          ema_period=20, 
                                                          atr_period=20, 
                                                          multiplier=2.5)
    
    # Volume analysis
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # ATR for stop loss
    atr_20 = calculate_atr(high, low, close, period=20)
    
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
    
    warmup = 80  # Need 50 for 1d SMA + 20 for Keltner + buffer
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(kelt_upper[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND FILTER ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === KELTNER BREAKOUT DETECTION ===
        # Price breaks above upper band = bullish breakout
        breakout_above = close[i] > kelt_upper[i]
        # Price breaks below lower band = bearish breakout
        breakout_below = close[i] < kelt_lower[i]
        
        # Price within bands = no breakout
        within_bands = not breakout_above and not breakout_below
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.4
        
        # === MINIMUM HOLD: 2 bars (12h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === ATR TRAILING STOP ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                # Long stop: price dropped 2.5 ATR from highest
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                # Short stop: price rose 2.5 ATR from lowest
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit if trend reverses and we've held minimum
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
            # LONG: Breakout above Keltner upper + volume + uptrend
            if breakout_above and vol_confirm and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_20[i] if not np.isnan(atr_20[i]) else atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG WEAK: Breakout but no volume (partial position)
            elif breakout_above and htf_bullish and not vol_confirm:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_20[i] if not np.isnan(atr_20[i]) else atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.5  # Half size without volume confirm
            
            # SHORT: Breakout below Keltner lower + volume + downtrend
            elif breakout_below and vol_confirm and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_20[i] if not np.isnan(atr_20[i]) else atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT WEAK: Breakout but no volume (partial position)
            elif breakout_below and htf_bearish and not vol_confirm:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_20[i] if not np.isnan(atr_20[i]) else atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.5  # Half size without volume confirm
            
            else:
                signals[i] = 0.0
    
    return signals