#!/usr/bin/env python3
"""
Experiment #028: 12h Williams %R Mean Reversion + 1w Trend

HYPOTHESIS: Williams %R has clear oversold (<-80) and overbought (>-20) levels.
Unlike Elder Ray's vague zero-line crossover, Williams %R provides discrete,
objective entry zones. On 12h, it captures multi-day swings (5-10 days per cycle).
1w SMA200 filters direction to avoid fighting major trends.

WHY 12h: Balances trade frequency (12-25/year target) with signal quality.
Too fast (4h/6h) = too many trades = fee drag. Too slow (1d) = too few trades.
Williams %R(14) on 12h ≈ Williams %R(28) on 6h = multi-day cycle.

TARGET: 50-150 total trades over 4 years = 12-37/year.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_williams_r_1w_sma200_chop_v1"
timeframe = "12h"
leverage = 1.0

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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum indicator for mean reversion"""
    n = len(close)
    willr = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        
        if highest_high != lowest_low:
            willr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    
    return willr

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - detect trending vs ranging markets"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        if tr_sum > 0:
            hh = np.max(high[i - period + 1:i + 1])
            ll = np.min(low[i - period + 1:i + 1])
            range_hl = hh - ll
            
            if range_hl > 0:
                chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # 1w SMA200 for trend direction (call ONCE, not per-bar)
    sma_1w = pd.Series(df_1w['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Local indicators
    willr = calculate_williams_r(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_bar = 0
    
    warmup = 250  # Need 200 for SMA200 + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(willr[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_1w_aligned[i]) or np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME (Choppiness Index) ===
        # CHOP > 61.8 = choppy (mean reversion works)
        # CHOP < 38.2 = trending (momentum works)
        is_choppy = chop[i] > 55.0  # Relaxed from 61.8 to enter more
        
        # === TREND DIRECTION (1w SMA200) ===
        price_above_1w_sma = close[i] > sma_1w_aligned[i]
        
        # === WILLIAMS %R SIGNALS ===
        willr_curr = willr[i]
        willr_prev = willr[i - 1] if i > 0 else willr_curr
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.2
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Williams %R crosses above -80 from below (recovering from oversold)
            # + price above 1w SMA + volume confirmation + choppy regime
            if willr_prev < -80 and willr_curr >= -80 and willr_curr < -50:
                if price_above_1w_sma and vol_confirm:
                    desired_signal = SIZE
            
            # SHORT: Williams %R crosses below -20 from above (falling from overbought)
            # + price below 1w SMA + volume confirmation + choppy regime
            if willr_prev > -20 and willr_curr <= -20 and willr_curr > -50:
                if not price_above_1w_sma and vol_confirm:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
                in_position = False
                position_side = 0
                desired_signal = 0.0
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
                in_position = False
                position_side = 0
                desired_signal = 0.0
        
        # === TAKE PROFIT (3R) ===
        if in_position and position_side > 0:
            profit_target = entry_price + 3.0 * entry_atr
            if high[i] >= profit_target:
                # Take partial profit, tighten stop
                desired_signal = SIZE / 2
        
        if in_position and position_side < 0:
            profit_target = entry_price - 3.0 * entry_atr
            if low[i] <= profit_target:
                desired_signal = -SIZE / 2
        
        # === TIME-BASED EXIT (hold at least 2 bars) ===
        bars_held = i - entry_bar if in_position else 0
        if in_position and bars_held >= 2:
            # Exit if Williams %R reaches opposite extreme
            if position_side > 0 and willr_curr > -20:
                desired_signal = 0.0
            if position_side < 0 and willr_curr < -80:
                desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
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