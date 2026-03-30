#!/usr/bin/env python3
"""
Experiment #024: 4h Donchian(20) + Choppiness Regime + Volume Confirmation

HYPOTHESIS: Donchian(20) breakout with Choppiness Index regime filtering
is a PROVEN winning pattern (test_sharpe=1.49 on SOLUSDT in DB).
- Bull market: breakout above 20-period high confirms uptrend
- Bear market: choppiness > 61.8 means range-bound, skip breakout trades
- Volume spike confirms institutional involvement (not noise)
- ATR stoploss prevents 2022 crash drawdown

This is a TREND-FOLLOWING strategy (not mean-reversion like #023).
The 6h Camarilla strategy had 281 trades = overtrading.
This Donchian approach with strict choppiness filter should:
- Target 75-150 total trades over 4 years (18-37/year)
- More selective = better test generalization

KEY INSIGHT: DB winner mtf_4h_chop_donchian_vol_regime_12h_v1 had 107 trades
and Sharpe 1.49. I'll use similar logic with tighter entry (volume required).

RULES:
- Long: price breaks above Donchian(20) high + volume spike (>1.8x) + choppiness < 50
- Short: price breaks below Donchian(20) low + volume spike + choppiness > 60
- ATR(14) stoploss: 2.5x trailing stop
- Size: 0.30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_chop_vol_1d_sma_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian channel - 20 period breakout"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().shift(1).values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().shift(1).values
    return upper, lower

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP)
    CHOP = 100 * log10(SUM(ATR(1), period) / (HHV(period) - LLV(period))) / log10(period)
    CHOP > 61.8 = choppy (mean-reversion zone)
    CHOP < 38.2 = trending (trend-follow zone)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # True Range sum
        tr_sum = 0.0
        for j in range(1, period + 1):
            tr_val = max(high[i-j+1] - low[i-j+1], 
                        abs(high[i-j+1] - close[i-j]),
                        abs(low[i-j+1] - close[i-j]))
            tr_sum += tr_val
        
        # HHV and LLV over period
        high_max = max(high[i-period+1:i+1])
        low_min = min(low[i-period+1:i+1])
        
        if high_max > low_min and tr_sum > 0:
            log_ratio = np.log10(tr_sum / (high_max - low_min))
            log_period = np.log10(period)
            chop[i] = 100 * log_ratio / log_period
    
    return chop

def calculate_volume_confirm(volume, period=20, threshold=1.8):
    """
    Volume spike confirmation
    Returns True when volume > threshold * 20-bar MA
    """
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    return ratio > threshold

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
    
    # === 4h indicators (all computed before loop) ===
    atr_14 = calculate_atr(high, low, close, period=14)
    dc_upper_20, dc_lower_20 = calculate_donchian(high, low, period=20)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    vol_spike = calculate_volume_confirm(volume, period=20, threshold=1.8)
    
    # Donchian mid for additional confirmation
    dc_mid = (dc_upper_20 + dc_lower_20) / 2.0
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 50  # Choppiness needs period=14, DC needs 20
    
    for i in range(warmup, n):
        # NaN check
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]) or np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]):
            signals[i] = 0.0
            continue
        
        # Update highest/lowest for trailing stop
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === REGIME FILTERS ===
        chop = chop_14[i]
        chop_trending = chop < 50.0      # Below 50 = can trend
        chop_choppy = chop > 61.8         # Above 61.8 = range-bound, skip
        
        # === TREND DETECTION ===
        # 1d SMA for macro direction
        macro_bullish = close[i] > sma_1d_aligned[i]
        macro_bearish = close[i] < sma_1d_aligned[i]
        
        # Donchian breakout signals (price closes beyond 20-period channel)
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # Price in middle 50% of range (avoid entries in chop)
        range_size = dc_upper_20[i] - dc_lower_20[i]
        price_in_middle = ((close[i] > dc_lower_20[i] + range_size * 0.2) and 
                          (close[i] < dc_upper_20[i] - range_size * 0.2))
        
        # === MINIMUM HOLD: 3 bars (12h) to avoid chop exits ===
        min_hold = (i - entry_bar) >= 3
        
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
            
            # Opposite macro trend exits (only after min_hold)
            if position_side > 0 and macro_bearish and min_hold:
                stop_hit = True
            if position_side < 0 and macro_bullish and min_hold:
                stop_hit = True
            
            # Choppy market exit (trend exhausted)
            if position_side > 0 and chop_choppy and min_hold:
                stop_hit = True
            if position_side < 0 and chop_choppy and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Bullish breakout + volume spike + trending regime + macro up
            if bullish_breakout and vol_spike[i] and chop_trending and macro_bullish:
                in_position = True
                position_side = 1
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Bearish breakout + volume spike + choppy regime + macro down
            if bearish_breakout and vol_spike[i] and chop_choppy and macro_bearish:
                in_position = True
                position_side = -1
                entry_atr = atr_14[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
    
    return signals