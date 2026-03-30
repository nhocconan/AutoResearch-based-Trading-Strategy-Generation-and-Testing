#!/usr/bin/env python3
"""
Experiment #022: 1w Supertrend + EMA + WilliamsR + Volume

HYPOTHESIS: Weekly timeframe with multiple confirmations should:
- Generate 40-80 trades over 4 years (enough for statistical validity)
- Weekly Supertrend catches major trend reversals
- Weekly EMA(13) filters for macro trend direction
- Williams %R catches momentum extremes (overbought/oversold)
- Volume spike confirms breakout strength
- Should work in both 2021 bull (+200% BTC) and 2022 bear (-77% BTC)

WHY 1w AS PRIMARY:
- Weekly bars = 208 total over 4 years
- Each trade = 2-4 weeks average holding
- Target: 40-80 trades = 10-20/year
- Much lower noise than 4h/1d, better signal quality

TRADE COUNT: 40-80 total over 4 years (10-20/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1w_supertrend_ema_willr_vol_v1"
timeframe = "1w"
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

def supertrend(high, low, close, period=7, multiplier=2.5):
    """
    Supertrend indicator
    Returns: supertrend values (positive = bullish, negative = bearish)
    """
    atr = calculate_atr(high, low, close, period)
    n = len(close)
    
    # Upper and Lower bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supert = np.zeros(n)
    direction = np.zeros(n)  # 1 = uptrend, -1 = downtrend
    
    supert[0] = lower_band[0]
    direction[0] = 1
    
    for i in range(1, n):
        # Calculate supertrend
        if close[i] > upper_band[i-1]:
            direction[i] = 1
            supert[i] = lower_band[i]
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
            supert[i] = upper_band[i]
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1:
                supert[i] = max(lower_band[i], supert[i-1])
            else:
                supert[i] = min(upper_band[i], supert[i-1])
    
    # Return signed supertrend (positive = above = bullish, negative = below = bearish)
    return direction * supert

def williams_r(high, low, close, period=14):
    """Williams %R"""
    n = len(close)
    wr = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            wr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return wr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Weekly indicators (primary TF) ===
    # Supertrend(8, 2.5) - captures weekly trend reversals
    st_values = supertrend(high, low, close, period=8, multiplier=2.5)
    st_direction = np.sign(st_values)  # 1 = bullish, -1 = bearish
    
    # Williams %R(14) on weekly - momentum oscillator
    willr = williams_r(high, low, close, period=14)
    
    # Volume: 20-week MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_ratio = volume / np.where(vol_ma > 1e-10, vol_ma, 1.0)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, period=14)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative for weekly (larger moves)
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    prev_st_direction = 0
    
    warmup = 30  # Williams %R needs 14 + buffer
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(st_direction[i]) or np.isnan(willr[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === Update highest/lowest for trailing stop ===
        if in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
        
        # === Supertrend flip detection ===
        st_flip_up = (prev_st_direction < 0) and (st_direction[i] > 0)
        st_flip_down = (prev_st_direction > 0) and (st_direction[i] < 0)
        
        # === Williams %R extremes ===
        willr_oversold = willr[i] < -80  # Strong bullish momentum
        willr_overbought = willr[i] > -20  # Strong bearish momentum
        
        # === Volume spike (>1.5x average) ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ATR TRAILING STOP (2.5x ATR from entry high/low) ===
        def check_atr_stop():
            if not in_position:
                return False
            if position_side > 0:
                return low[i] < (highest_since_entry - 2.5 * entry_atr)
            else:
                return high[i] > (lowest_since_entry + 2.5 * entry_atr)
        
        # === MINIMUM HOLD: 2 bars (2 weeks) to avoid immediate reversals ===
        min_hold_bars = 2
        min_hold = (i - entry_bar) >= min_hold_bars
        
        # === EXITS ===
        if in_position:
            stop_hit = check_atr_stop()
            
            # Exit on trend reversal (Supertrend flips opposite)
            if position_side > 0 and st_direction[i] < 0 and min_hold:
                stop_hit = True
            if position_side < 0 and st_direction[i] > 0 and min_hold:
                stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: Supertrend flips bullish + Williams %R oversold + volume spike
            if st_flip_up and willr_oversold and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
            
            # SHORT: Supertrend flips bearish + Williams %R overbought + volume spike
            elif st_flip_down and willr_overbought and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            
            else:
                signals[i] = 0.0
        
        # Update previous direction for flip detection
        prev_st_direction = st_direction[i]
    
    return signals