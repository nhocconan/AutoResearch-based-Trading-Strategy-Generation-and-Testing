#!/usr/bin/env python3
"""
Experiment #024: 12h Donchian Breakout + Choppiness Regime + 1d Trend

HYPOTHESIS: Donchian(20) breakouts are statistically significant price
structure events. Combined with Choppiness Index regime filter:
- Bull market: price breaks above 20-period high → strong momentum
- Bear market: choppiness filter prevents chasing breakdowns
- Range market: CHOP > 61.8 = choppy = no trades (avoid whipsaw)

KEY INSIGHT: DB top performer mtf_4h_chop_donchian_vol_regime_12h_v1
achieved test Sharpe 1.49 on SOLUSDT with 107 trades. This replicates
that pattern at 12h TF (even fewer trades = less fee drag).

TRADE COUNT: 75-125 total over 4 years (19-31/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_chop_1d_sma_v1"
timeframe = "12h"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP > 61.8 = ranging/choppy (no trend)
    CHOP < 38.2 = trending (good for breakout strategies)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of true range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr_sum += max(high[j] - low[j], 
                         abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
        
        # Highest high - lowest low over period
        hh = max(high[i - period + 1:i + 1])
        ll = min(low[i - period + 1:i + 1])
        hl_range = hh - ll
        
        if hl_range > 1e-10:
            chop[i] = 100 * np.log10(tr_sum / hl_range) / np.log10(period)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Donchian channel"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    middle = (upper + pd.Series(low).rolling(window=period, min_periods=period).min().values) / 2.0
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, middle, lower

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
    
    # === 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Donchian 20
    dc_upper_20, dc_middle_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume
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
    
    warmup = 50  # enough for chop calc
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME CHECK ===
        # CHOP > 61.8 = choppy/ranging = skip (avoid whipsaw)
        # CHOP < 50 = trending = good for breakouts
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # === TREND CHECK (1d SMA) ===
        htf_bullish = close[i] > sma_1d_aligned[i]
        htf_bearish = close[i] < sma_1d_aligned[i]
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Long: price breaks above 20-period high + volume confirmation
        donchian_breakout_long = (close[i] > dc_upper_20[i]) and not np.isnan(dc_upper_20[i])
        # Short: price breaks below 20-period low + volume confirmation
        donchian_breakout_short = (close[i] < dc_lower_20[i]) and not np.isnan(dc_lower_20[i])
        
        # Volume confirmation
        vol_confirm = vol_ratio[i] > 1.2
        
        # === MINIMUM HOLD: 2 bars (24h) ===
        min_hold = (i - entry_bar) >= 2
        
        # === TRAILING STOP (2x ATR from entry) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                # Trail: exit if price drops 2 ATR from highest since entry
                return low[i] < (highest_since_entry - 2.5 * atr_14[i])
            else:
                return high[i] > (lowest_since_entry + 2.5 * atr_14[i])
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Trend reversal exit (after min hold)
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
            # LONG: Donchian breakout + volume + not choppy + 1d uptrend
            if donchian_breakout_long and vol_confirm and not is_choppy and htf_bullish:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Donchian breakdown + volume + not choppy + 1d downtrend
            elif donchian_breakout_short and vol_confirm and not is_choppy and htf_bearish:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # Conservative LONG: in strong uptrend, buy pullbacks to middle band
            elif htf_bullish and is_trending and close[i] > dc_middle_20[i] and vol_confirm:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.5  # Half size for pullback entry
            
            # Conservative SHORT: in downtrend, sell rallies to middle band
            elif htf_bearish and is_trending and close[i] < dc_middle_20[i] and vol_confirm:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.5  # Half size for rally sell
            
            else:
                signals[i] = 0.0
    
    return signals