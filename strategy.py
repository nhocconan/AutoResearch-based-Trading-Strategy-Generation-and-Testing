#!/usr/bin/env python3
"""
Experiment #024: 6h Vortex Indicator + 1d VWAP + Volume Confirmation

HYPOTHESIS: Vortex Indicator (VI) measures directional momentum by comparing
current bar ranges to historical ranges. VI+ > VI- = bullish flow, VI- > VI+ = bearish.
Combined with 1d VWAP as institutional anchor and volume confirmation:
- VI cross + price > 1d VWAP = long continuation in bull
- VI cross + price < 1d VWAP = short continuation in bear
- Range: both sides fade when no clear VI dominance

KEY INSIGHT: VI is a leading indicator unlike SMA/EMA. It catches momentum
shifts BEFORE price breaks structure. Works in both trending and choppy markets.

TARGET: 75-150 total trades over 4 years (18-37/year).
Size: 0.30.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_vortex_vwap_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_vortex(high, low, close, period=14):
    """
    Vortex Indicator (VT/VI)
    VM+ = |high_t - low_{t-1}|
    VM- = |low_t - high_{t-1}|
    TR = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
    VI+ = sum(VM+) / sum(TR) over period
    VI- = sum(VM-) / sum(TR) over period
    """
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    vm_plus = np.zeros(n, dtype=np.float64)
    vm_minus = np.zeros(n, dtype=np.float64)
    
    # First bar
    tr[0] = high[0] - low[0]
    vm_plus[0] = 0.0
    vm_minus[0] = 0.0
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
        vm_plus[i] = abs(high[i] - low[i-1])
        vm_minus[i] = abs(low[i] - high[i-1])
    
    # Rolling sums
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    vm_plus_sum = pd.Series(vm_plus).rolling(window=period, min_periods=period).sum().values
    vm_minus_sum = pd.Series(vm_minus).rolling(window=period, min_periods=period).sum().values
    
    # VI ratios
    vi_plus = np.where(tr_sum > 1e-10, vm_plus_sum / tr_sum, 0.0)
    vi_minus = np.where(tr_sum > 1e-10, vm_minus_sum / tr_sum, 0.0)
    
    return vi_plus, vi_minus

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

def calculate_vwap(high, low, close, volume):
    """
    Volume Weighted Average Price
    VWAP = sum(price * volume) / sum(volume)
    """
    typical = (high + low + close) / 3.0
    cum_vol = np.cumsum(volume)
    cum_pv = np.cumsum(typical * volume)
    vwap = cum_pv / np.where(cum_vol > 1e-10, cum_vol, 1.0)
    return vwap

def calculate_donchian(high, low, period=20):
    """Donchian channel for structure"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d VWAP for institutional anchor (call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate VWAP on 1d data
    htf_high = df_1d['high'].values
    htf_low = df_1d['low'].values
    htf_close = df_1d['close'].values
    htf_volume = df_1d['volume'].values
    htf_vwap = calculate_vwap(htf_high, htf_low, htf_close, htf_volume)
    
    # Align to 6h
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, htf_vwap)
    
    # === 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    vi_plus, vi_minus = calculate_vortex(high, low, close, period=14)
    
    # Donchian for structure
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    
    # Volume spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # VI signal line (VI+ - VI-)
    vi_diff = vi_plus - vi_minus
    
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
    
    warmup = 50  # Need 14 for VI + buffer
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(vi_plus[i]) or np.isnan(vwap_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === TREND DETECTION via VI cross ===
        # VI+ crosses above VI- = bullish momentum shift
        vi_bullish = vi_plus[i] > vi_minus[i]
        # VI- crosses above VI+ = bearish momentum shift
        vi_bearish = vi_minus[i] > vi_plus[i]
        
        # Macro trend via 1d VWAP
        above_vwap = close[i] > vwap_1d_aligned[i]
        below_vwap = close[i] < vwap_1d_aligned[i]
        
        # Structure: price in upper/lower third of Donchian
        dc_mid = (dc_upper_20[i] + dc_lower_20[i]) / 2.0 if not np.isnan(dc_upper_20[i]) else close[i]
        price_in_upper = close[i] > dc_mid
        price_in_lower = close[i] < dc_mid
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.4
        
        # === MINIMUM HOLD: 2 bars (12h) ===
        min_hold = (i - entry_bar) >= 2
        
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
            
            # VI reversal exits (when momentum shifts against position)
            if position_side > 0 and vi_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and vi_bullish and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: VI bullish cross + price above VWAP + volume spike
            if vi_bullish and above_vwap and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # LONG CONSERVATIVE: VI bullish + above VWAP (no volume req)
            elif vi_bullish and above_vwap and price_in_upper:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE * 0.7  # Smaller size without volume confirm
            
            # SHORT: VI bearish cross + price below VWAP + volume spike
            elif vi_bearish and below_vwap and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            # SHORT CONSERVATIVE: VI bearish + below VWAP (no volume req)
            elif vi_bearish and below_vwap and price_in_lower:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE * 0.7  # Smaller size without volume confirm
            
            else:
                signals[i] = 0.0
    
    return signals