#!/usr/bin/env python3
"""
Experiment #027: 12h Vortex Direction + ATR Stoploss + 1w Trend

HYPOTHESIS: Vortex Indicator (VI) is an underutilized trend detector that 
measures +VI vs -VI crossovers to signal trend changes. Unlike RSI/EMA, it 
captures structural momentum shifts. Combined with 1w HMA for trend bias 
and volume confirmation, this gives clean entries at trend reversals.

WHY IT WORKS IN BULL AND BEAR:
- Bull: +VI crosses above -VI → trend up → enter long
- Bear: -VI crosses above +VI → trend down → enter short  
- Range: both VI lines oscillate → no trades (avoid whipsaw)
- ATR stoploss protects against 2022-style crashes

TIMEFRAME: 12h primary
HTF: 1w for trend bias
TARGET: 75-150 total trades over 4 years (19-37/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_vortex_atr_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

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

def calculate_vortex(high, low, close, period=14):
    """
    Vortex Indicator - returns +VI and -VI
    +VI > -VI indicates uptrend
    -VI > +VI indicates downtrend
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    # True Range components
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Upward Movement (VM+)
    up_move = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        up_move[i] = abs(high[i] - low[i-1])
    
    # Downward Movement (VM-)
    dn_move = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        dn_move[i] = abs(low[i] - high[i-1])
    
    # Sum over period
    tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
    up_sum = pd.Series(up_move).rolling(window=period, min_periods=period).sum().values
    dn_sum = pd.Series(dn_move).rolling(window=period, min_periods=period).sum().values
    
    # VI lines
    vi_plus = np.full(n, np.nan, dtype=np.float64)
    vi_minus = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        if tr_sum[i] > 0:
            vi_plus[i] = up_sum[i] / tr_sum[i]
            vi_minus[i] = dn_sum[i] / tr_sum[i]
    
    return vi_plus, vi_minus

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w HMA for trend bias (bull if price > HMA)
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate local 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Vortex Indicator
    vi_plus, vi_minus = calculate_vortex(high, low, close, period=14)
    
    # Previous VI for crossover detection
    vi_plus_prev = np.roll(vi_plus, 1)
    vi_minus_prev = np.roll(vi_minus, 1)
    vi_plus_prev[0] = np.nan
    vi_minus_prev[0] = np.nan
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(vi_plus[i]) or np.isnan(vi_minus[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND BIAS (1w HMA) ===
        price_above_1w_hma = close[i] > hma_1w_aligned[i]
        hma_trend_bull = price_above_1w_hma
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.2
        
        # === VORTEX CROSSOVER DETECTION ===
        # +VI crosses above -VI = bullish crossover
        bullish_cross = (vi_plus[i] > vi_minus[i]) and (vi_plus_prev[i] <= vi_minus_prev[i] if not np.isnan(vi_plus_prev[i]) else True)
        # -VI crosses above +VI = bearish crossover
        bearish_cross = (vi_minus[i] > vi_plus[i]) and (vi_minus_prev[i] <= vi_plus_prev[i] if not np.isnan(vi_minus_prev[i]) else True)
        
        # VI strength (how far apart are they?)
        vi_spread = abs(vi_plus[i] - vi_minus[i])
        vi_strong = vi_spread > 0.1  # meaningful trend when spread > 0.1
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === NEW LONG ENTRY ===
            # +VI crosses above -VI + bullish 1w trend + volume confirmation
            if bullish_cross and hma_trend_bull and (vol_spike or vi_strong):
                desired_signal = SIZE
            
            # === NEW SHORT ENTRY ===
            # -VI crosses above +VI + bearish 1w trend + volume confirmation
            if bearish_cross and not hma_trend_bull and (vol_spike or vi_strong):
                desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TRAILING STOP PROFIT PROTECTION ===
        if in_position and position_side > 0:
            # If up 2R, tighten stop to entry
            if close[i] > entry_price + 2.0 * entry_atr:
                new_stop = entry_price + 0.5 * entry_atr  # lock in 1.5R
                stop_price = max(stop_price, new_stop)
                if low[i] < stop_price:
                    stoploss_triggered = True
                    desired_signal = 0.0
        
        if in_position and position_side < 0:
            if close[i] < entry_price - 2.0 * entry_atr:
                new_stop = entry_price - 0.5 * entry_atr
                stop_price = min(stop_price, new_stop)
                if high[i] > stop_price:
                    stoploss_triggered = True
                    desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
            else:
                # Same direction - maintain position
                pass
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals