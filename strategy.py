#!/usr/bin/env python3
"""
Experiment #021: 12h Aroon + Bollinger Band Position + 1d SMA200 Trend

HYPOTHESIS: Aroon crossover (period=25) gives clean, reliable trend shifts 
without the noise of RSI or stochastic. Combined with 1d SMA200 trend alignment 
and Bollinger Band position for entry timing, this catches major trend changes.

WHY IT WORKS IN BULL AND BEAR: Aroon measures time since highs/lows.
Bull: Aroon Up stays elevated. Bear: Aroon Down stays elevated.
Crossover signals regime change - symmetric in both directions.

WHY 12h: Slow enough for meaningful Aroon signals (25-period = 12.5 days).
Faster than 12h = too many false crossovers. 12h = 2-3 major signals/month.

TARGET: 40-100 total trades over 4 years = 10-25/year.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_aroon_bb_sma200_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_aroon(high, low, period=25):
    """Aroon indicator - returns Aroon Up and Aroon Down"""
    n = len(high)
    aroon_up = np.zeros(n, dtype=np.float64)
    aroon_down = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        # Count bars since highest high in period
        max_idx = np.argmax(high[i-period:i+1])
        bars_since_high = period - max_idx
        aroon_up[i] = (period - bars_since_high) / period * 100
        
        # Count bars since lowest low in period
        min_idx = np.argmin(low[i-period:i+1])
        bars_since_low = period - min_idx
        aroon_down[i] = (period - bars_since_low) / period * 100
    
    return aroon_up, aroon_down

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
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction (used as filter, not entry)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Aroon (25-period = 12.5 days on 12h)
    aroon_up, aroon_down = calculate_aroon(high, low, period=25)
    
    # Bollinger Bands (20, 2.0) for entry timing
    bb_sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_sma + 2.0 * bb_std
    bb_lower = bb_sma - 2.0 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_sma
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    aroon_flipped = False  # Track if we need new Aroon crossover
    
    warmup = 300  # Need 200 for SMA200 alignment buffer + 25 for Aroon
    
    for i in range(warmup, n):
        # Skip if ATR not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # Skip if SMA200 not aligned
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        
        # === AROON CROSSOVER (trend shift signal) ===
        aroon_cross_up = aroon_up[i] > aroon_down[i] and aroon_up[i-1] <= aroon_down[i-1]
        aroon_cross_down = aroon_down[i] > aroon_up[i] and aroon_down[i-1] <= aroon_up[i-1]
        
        # Bollinger Band position (0=lower, 50=middle, 100=upper)
        if bb_std[i] > 0:
            bb_pos = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) * 100
        else:
            bb_pos = 50
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.3
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # Reset flip flag when flat
            aroon_flipped = False
            
            # === LONG: Aroon bullish crossover + price above 1d SMA200 + BB near lower ===
            if price_above_1d_sma and aroon_cross_up:
                # BB position < 40 means price is near lower band (good entry)
                if bb_pos < 45:
                    desired_signal = SIZE
                    aroon_flipped = True
            
            # === SHORT: Aroon bearish crossover + price below 1d SMA200 + BB near upper ===
            if not price_above_1d_sma and aroon_cross_down:
                # BB position > 60 means price is near upper band (good entry)
                if bb_pos > 55:
                    desired_signal = -SIZE
                    aroon_flipped = True
        
        # === STOPLOSS (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                desired_signal = 0.0
        
        # === TAKE PROFIT (3:1 ratio or reverse Aroon signal) ===
        if in_position and position_side > 0:
            profit_target = entry_atr * 3.0
            if close[i] >= close[i-1] + profit_target:
                # 3:1 achieved, take profit
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            profit_target = entry_atr * 3.0
            if close[i] <= close[i-1] - profit_target:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = close[i] - 2.5 * entry_atr
                else:
                    stop_price = close[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                stop_price = 0.0
        
        signals[i] = desired_signal
    
    return signals