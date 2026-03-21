#!/usr/bin/env python3
"""
Experiment #334: 4h KAMA Adaptive Trend + Fisher Transform Entries + Daily HMA Bias
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility better than HMA/EMA.
Combined with Fisher Transform for precise entry timing (catches reversals at extremes),
and daily HMA for macro trend bias. Fisher Transform normalizes price to Gaussian distribution,
making extreme crosses (-2/+2) reliable reversal signals. Volume confirmation reduces false breakouts.
Timeframe: 4h (REQUIRED), HTF: 1d for trend bias via mtf_data helper.
Target: Beat Sharpe=0.499 with adaptive trend following + precise Fisher entries.
Key insight: KAMA adapts to choppy vs trending markets automatically, Fisher gives clean entry signals.
Position sizing: 0.25 entry, 0.125 half (take profit), stoploss at 2.5*ATR.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_daily_hma_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts smoothing based on market efficiency (trend vs noise).
    ER (Efficiency Ratio) = |net change| / sum of absolute changes
    High ER = trending (use fast SC), Low ER = noisy (use slow SC)
    """
    close_s = pd.Series(close)
    er = np.zeros(len(close))
    kama = np.zeros(len(close))
    
    # Calculate Efficiency Ratio
    for i in range(er_period, len(close)):
        net_change = np.abs(close[i] - close[i - er_period])
        sum_changes = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
        if sum_changes > 0:
            er[i] = net_change / sum_changes
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = np.square(er) * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    # Calculate KAMA
    for i in range(er_period + 1, len(close)):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price to Gaussian-normalized values for clearer reversal signals.
    Entry: Fisher crosses above -2 (long), crosses below +2 (short).
    """
    hl2 = (high + low) / 2
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize price to 0-1 range
    range_val = highest - lowest
    range_val = np.where(range_val == 0, 0.001, range_val)  # Avoid division by zero
    normalized = (hl2 - lowest) / range_val
    normalized = np.clip(normalized, 0.001, 0.999)  # Clamp for log calculation
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    fisher = pd.Series(fisher).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return fisher

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for faster trend response."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.where(np.isnan(vol_ratio), 1.0, vol_ratio)
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    kama_fast = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_slow = calculate_kama(close, er_period=20, fast_period=5, slow_period=50)
    fisher = calculate_fisher_transform(high, low, 9)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):  # Start after 250 bars for indicators
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(kama_fast[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # Daily macro trend bias
        daily_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        daily_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # KAMA trend state
        kama_bullish = kama_fast[i] > kama_slow[i]
        kama_bearish = kama_fast[i] < kama_slow[i]
        
        # KAMA crossover signals
        kama_cross_long = kama_fast[i] > kama_slow[i] and kama_fast[i-1] <= kama_slow[i-1]
        kama_cross_short = kama_fast[i] < kama_slow[i] and kama_fast[i-1] >= kama_slow[i-1]
        
        # Fisher Transform signals
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        fisher_cross_up = fisher[i] > -1.8 and fisher[i-1] <= -1.8
        fisher_cross_down = fisher[i] < 1.8 and fisher[i-1] >= 1.8
        
        # Volume confirmation
        volume_ok = vol_ratio[i] > 0.8  # At least 80% of average volume
        
        new_signal = 0.0
        
        # === LONG ENTRIES ===
        # Primary: KAMA crossover + Daily bullish + Fisher confirmation + Volume
        if kama_cross_long and daily_bullish and fisher_oversold and volume_ok:
            new_signal = SIZE_ENTRY
        # Secondary: KAMA bullish + Fisher cross up + Daily bullish
        elif kama_bullish and fisher_cross_up and daily_bullish and volume_ok:
            new_signal = SIZE_ENTRY
        # Tertiary: KAMA crossover without daily filter (momentum only)
        elif kama_cross_long and fisher[i] < -1.0 and volume_ok:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES ===
        # Primary: KAMA crossover + Daily bearish + Fisher confirmation + Volume
        if kama_cross_short and daily_bearish and fisher_overbought and volume_ok:
            new_signal = -SIZE_ENTRY
        # Secondary: KAMA bearish + Fisher cross down + Daily bearish
        elif kama_bearish and fisher_cross_down and daily_bearish and volume_ok:
            new_signal = -SIZE_ENTRY
        # Tertiary: KAMA crossover without daily filter (momentum only)
        elif kama_cross_short and fisher[i] > 1.0 and volume_ok:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from highest)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR from lowest)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals