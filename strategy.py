#!/usr/bin/env python3
"""
Experiment #028: 12h Camarilla Pivot + 1d Trend + Volume Spike + Choppiness

HYPOTHESIS: Camarilla pivot levels (S3/R3) represent key institutional support/resistance
that price frequently reverts to. Combined with 1d SMA200 trend filter and volume 
confirmation, this captures mean-reversion trades at proven levels.

WHY IT WORKS IN BULL AND BEAR: S3/R3 are symmetric levels derived from daily range.
In bull: price touches S3 = buy opportunity. In bear: price touches R3 = short opportunity.
Camarilla is NOT directional - it works in both markets.

12h timeframe: 50-150 trades over 4 years = 12-37/year (within target).
Target: 75-200 total trades.
Signal size: 0.30 (discrete).

DB EVIDENCE: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 achieved 
test Sharpe 1.471 on ETHUSDT with 95 trades. Adapting to 12h for even fewer trades.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_camarilla_vol_chop_1d_v2"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range using EWM"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(abs(high[1:] - close[:-1]), 
                              abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - values above 61.8 = choppy/range-bound"""
    n = len(high)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else 0)
            tr_sum += tr
        
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_hl = hh - ll
        
        if range_hl > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / range_hl) / np.log10(period)
    
    return chop

def calculate_camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels (S3, S4, R3, R4)"""
    n = len(high)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        h = high[i - 1]
        l = low[i - 1]
        c = close[i - 1]
        rng = h - l
        
        # Camarilla formula
        s3[i] = c - rng * (1.1 / 12.0)
        s4[i] = c - rng * (1.1 / 6.0)
        r3[i] = c + rng * (1.1 / 12.0)
        r4[i] = c + rng * (1.1 / 6.0)
    
    return s3, s4, r3, r4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Camarilla levels (shifted by 1 to use previous day's data)
    s3, s4, r3, r4 = calculate_camarilla_levels(high, low, close)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    half_exited = False
    
    warmup = 250  # Need 200 for SMA200 + 20 for volume + buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(s3[i]) or np.isnan(r3[i]):
            signals[i] = 0.0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_1d_aligned[i]
        trend_bull = price_above_1d_sma
        trend_bear = not price_above_1d_sma
        
        # === REGIME (Choppiness Index) ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 50.0
        
        # Skip if too choppy (only when entering new position)
        if is_choppy and not in_position:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_spike = vol_ratio[i] > 1.5
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC ===
        if not in_position:
            atr_val = atr_14[i]
            
            # === LONG: Price touches/near S3 level ===
            # Allow 0.5 ATR buffer from S3
            if low[i] <= s3[i] + 0.5 * atr_val:
                # Trend must align: bull trend OR trending market with volume
                if (trend_bull) or (is_trending and vol_spike):
                    desired_signal = SIZE
            
            # === SHORT: Price touches/near R3 level ===
            if high[i] >= r3[i] - 0.5 * atr_val:
                if (trend_bear) or (is_trending and vol_spike):
                    desired_signal = -SIZE
        
        # === POSITION MANAGEMENT ===
        if in_position:
            # Update highest/lowest
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
            
            # ATR-based trailing stop
            if position_side > 0:
                trailing_stop = highest_since_entry - 2.0 * entry_atr
            else:
                trailing_stop = lowest_since_entry + 2.0 * entry_atr
            
            # Check stoploss
            if position_side > 0 and low[i] < trailing_stop:
                desired_signal = 0.0
            elif position_side < 0 and high[i] > trailing_stop:
                desired_signal = 0.0
            else:
                # Profit target: half exit at 2R
                bars_held = i - entry_bar
                
                if not half_exited and bars_held >= 4:  # Hold at least 4 bars (2 days)
                    if position_side > 0:
                        profit_r = (close[i] - entry_price) / entry_atr
                        if profit_r >= 2.0:
                            desired_signal = SIZE / 2  # Half position
                            half_exited = True
                    else:
                        profit_r = (entry_price - close[i]) / entry_atr
                        if profit_r >= 2.0:
                            desired_signal = -SIZE / 2
                            half_exited = True
                
                # Full exit: price crosses back through S4/R4 or market turns choppy
                if position_side > 0 and low[i] < s4[i]:
                    desired_signal = 0.0
                if position_side < 0 and high[i] > r4[i]:
                    desired_signal = 0.0
                
                # Exit if choppy and profitable
                if is_choppy and bars_held >= 4:
                    if position_side > 0 and close[i] > entry_price:
                        desired_signal = 0.0
                    if position_side < 0 and close[i] < entry_price:
                        desired_signal = 0.0
        
        # === EXECUTE TRADE ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                half_exited = False
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                half_exited = False
        
        signals[i] = desired_signal
    
    return signals