#!/usr/bin/env python3
"""
Experiment #003: 4h Momentum Breakout with Williams %R Confirmation

HYPOTHESIS: Crypto markets make large directional moves with volume surges.
By entering on momentum breakouts (yesterday's high/low) WITH Williams %R confirming
momentum AND volume spike confirmation, we capture the start of trends while
filtering false breakouts.

WHY IT WORKS IN BULL AND BEAR:
- Bull: Long breakouts above yesterday's high when momentum aligns
- Bear: Short breakouts below yesterday's low when momentum aligns
- Symmetrical approach captures both directions

WHY WILLIAMS %R: More sensitive than RSI for entry timing. Enters when momentum
is strong but not overbought, exits before exhaustion.

DIFFERENTIATION FROM RECENT FAILURES:
- #017: Used HMA + Donchian = 418 trades (overtrading)
- #012/#016: Used Camarilla or ATR expansion = too few trades
- THIS: Williams %R + volume surge + breakout = targeted momentum entries
  Should get 100-150 total trades (balanced)

TARGET: 100-150 total over 4 years. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_willr_momentum_breakout_vol_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(abs(high[1:] - close[:-1]), 
                              abs(low[1:] - close[:-1])))
    tr = np.concatenate([[tr[0]], tr])
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    if n < period:
        return np.full(n, -50.0)
    
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    willr = np.where(highest - lowest > 0,
                     -100 * (highest - close) / (highest - lowest),
                     -50.0)
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA21 for trend direction
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    willr_14 = calculate_williams_r(high, low, close, period=14)
    
    # Volume ratio (20-bar)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    last_close_bar = 0
    
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if EMA not aligned
        if np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === HTF TREND DIRECTION (1d EMA21) ===
        price_above_1d_ema = close[i] > ema_1d_aligned[i]
        
        # === MOMENTUM (Williams %R) ===
        willr = willr_14[i]
        willr_bullish = willr < -50   # Not oversold (for longs)
        willr_bearish = willr > -50    # Not overbought (for shorts)
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === PRICE LEVELS from PREVIOUS bar (no look-ahead) ===
        prev_high = high[i - 1]
        prev_low = low[i - 1]
        prev_close = close[i - 1]
        
        # === YESTERDAY'S RANGE reference (2 bars ago for 4h) ===
        ref_high = high[i - 2] if i >= 2 else prev_high
        ref_low = low[i - 2] if i >= 2 else prev_low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above recent high + momentum + volume ===
            if price_above_1d_ema and willr_bullish and vol_spike:
                # Breakout above reference high (or prev_high)
                if close[i] > ref_high:
                    desired_signal = SIZE
                # Also enter on strong close above prev_high with volume
                elif close[i] > prev_high:
                    desired_signal = SIZE
            
            # === SHORT: Breakdown below recent low + momentum + volume ===
            if not price_above_1d_ema and willr_bearish and vol_spike:
                # Breakdown below reference low (or prev_low)
                if close[i] < ref_low:
                    desired_signal = -SIZE
                # Also enter on strong close below prev_low with volume
                elif close[i] < prev_low:
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
                last_close_bar = i
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
                last_close_bar = i
        
        # === MINIMUM HOLD: 2 bars to avoid fee churn ===
        bars_held = i - entry_bar
        if in_position and bars_held >= 2:
            # Check for reversal signal (opposite momentum)
            if position_side > 0 and willr_14[i] > -20:  # Getting overbought
                # Take profit on momentum reversal
                if close[i] > entry_price + 1.5 * entry_atr:
                    desired_signal = 0.0
            
            if position_side < 0 and willr_14[i] < -80:   # Getting oversold
                # Take profit on momentum reversal
                if close[i] < entry_price - 1.5 * entry_atr:
                    desired_signal = 0.0
        
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
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals