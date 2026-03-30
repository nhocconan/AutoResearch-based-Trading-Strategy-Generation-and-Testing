#!/usr/bin/env python3
"""
Experiment #023: 12h Williams Alligator Breakout + Volume + 1w EMA

HYPOTHESIS: Williams Alligator (SMMA-based trend system) captures institutional
trend starts. When price closes outside all 3 lines (Jaws/Teeth/Lips) AND
the lines are spreading apart = strong momentum breakout. Weekly EMA21 filters
direction. Volume confirms institutional participation.

WHY IT WORKS IN BULL AND BEAR: Symmetrical - long breakouts above Alligator
in uptrends, short breakouts below in downtrends. Alligator self-adjusts to
volatility. Works in both directions.

TARGET: 75-150 total trades over 4 years = 19-37/year. HARD MAX: 200.
Signal size: 0.25.

Williams Alligator formulas:
- Jaws = SMMA(close, 13) shifted 8 bars right
- Teeth = SMMA(close, 8) shifted 5 bars right  
- Lips = SMMA(close, 5) shifted 3 bars right
- Long: price above all 3 AND lips > teeth > jaws (spreading up)
- Short: price below all 3 AND lips < teeth < jaws (spreading down)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_alligator_breakout_vol_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_smma(data, period):
    """Smoothed Moving Average (SMMA) - Williams Alligator base"""
    n = len(data)
    if n < period:
        return np.full(n, np.nan)
    
    result = np.zeros(n, dtype=np.float64)
    # First value is SMA
    result[period - 1] = np.mean(data[:period])
    
    # SMMA formula: (prev_smma * (period - 1) + current) / period
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + data[i]) / period
    
    return result

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA21 for trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Williams Alligator components
    # Jaws: SMMA(13) shifted 8 bars right (representing ~4 days on 12h)
    jaws_raw = calculate_smma(close, 13)
    teeth_raw = calculate_smma(close, 8)
    lips_raw = calculate_smma(close, 5)
    
    # Shift right (align to future for plotting, but we need current values)
    # Actually shift right by 8,5,3 to match Williams original
    jaws = np.full(n, np.nan)
    teeth = np.full(n, np.nan)
    lips = np.full(n, np.nan)
    
    # Shift by offset (bars ahead)
    jaws[8:] = jaws_raw[:-8]
    teeth[5:] = teeth_raw[:-5]
    lips[3:] = lips_raw[:-3]
    
    # Volume ratio (20-bar SMA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # ADX for trend strength (25 = start trending)
    # Simple ADX approximation using EMA of price momentum
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    plus_di = pd.Series(plus_dm / np.where(atr_14 > 0, atr_14, 1)).ewm(span=14, min_periods=14).mean().values
    minus_di = pd.Series(minus_dm / np.where(atr_14 > 0, atr_14, 1)).ewm(span=14, min_periods=14).mean().values
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14).mean().values
    
    # Signals
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
    bars_since_signal = 0  # Prevent immediate re-entry
    
    warmup = 50  # Alligator + alignment buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(ema_1w_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Need valid Alligator values
        if np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        bars_since_signal += 1
        
        # === TREND DIRECTION (1w EMA21) ===
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        
        # === ALLIGATOR STATE ===
        # Long setup: price above all lines AND lines spreading up (lips > teeth > jaws)
        alligator_long = (close[i] > jaws[i]) and (close[i] > teeth[i]) and (close[i] > lips[i])
        spreading_long = (lips[i] > teeth[i]) and (teeth[i] > jaws[i])
        
        # Short setup: price below all lines AND lines spreading down (lips < teeth < jaws)
        alligator_short = (close[i] < jaws[i]) and (close[i] < teeth[i]) and (close[i] < lips[i])
        spreading_short = (lips[i] < teeth[i]) and (teeth[i] < jaws[i])
        
        # Volume confirmation (moderate spike)
        vol_spike = vol_ratio[i] > 1.3
        
        # ADX trend strength (optional - loosens entry slightly)
        strong_trend = adx[i] > 20  # Lower threshold for more signals
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Breakout above Alligator + trend alignment ===
            # Require: above 1w EMA, price above all lines, spreading up, volume confirm
            if price_above_1w_ema and alligator_long and spreading_long and vol_spike:
                if bars_since_signal >= 3:  # Minimum 1.5 days between signals
                    desired_signal = SIZE
                    bars_since_signal = 0
            
            # === SHORT: Breakdown below Alligator + trend alignment ===
            if not price_above_1w_ema and alligator_short and spreading_short and vol_spike:
                if bars_since_signal >= 3:
                    desired_signal = -SIZE
                    bars_since_signal = 0
        
        # === STOPLOSS (2.0 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (3:1 RR or Alligator reversal) ===
        if in_position:
            bars_held = i - entry_bar
            
            # 3:1 profit target
            if position_side > 0:
                profit_target = entry_price + 3.0 * entry_atr
                if high[i] >= profit_target:
                    desired_signal = SIZE / 2  # Half position
            elif position_side < 0:
                profit_target = entry_price - 3.0 * entry_atr
                if low[i] <= profit_target:
                    desired_signal = -SIZE / 2  # Half position
            
            # Exit if Alligator flips (opposite spread)
            if position_side > 0 and spreading_short:
                desired_signal = 0.0
            if position_side < 0 and spreading_long:
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