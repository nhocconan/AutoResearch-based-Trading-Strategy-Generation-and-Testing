#!/usr/bin/env python3
"""
Experiment #022: 4h Camarilla Pivot + Volume Spike + Choppiness Regime

HYPOTHESIS: Camarilla pivot levels from the daily timeframe represent 
institutional support/resistance zones. Combined with volume confirmation 
and choppiness regime filtering (CHOP < 50 = trending, trade breakouts; 
CHOP > 61.8 = ranging, fade moves to S4/R4), this strategy captures 
high-probability mean reversion and trend continuation trades.

KEY INSIGHT from DB: gen_camarilla_pivot_volume_spike_choppiness_4h_v1 achieved 
test Sharpe=1.471 on ETHUSDT with 95 trades. This is the TOP performer.

TIMEFRAME: 4h primary
HTF: 1d for Camarilla levels
REGIME: Choppiness Index for trend/range detection
TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_vol_chop_v1"
timeframe = "4h"
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

def calculate_choppiness(high, low, close, period=14):
    """Choppiness Index - measures trend vs range"""
    n = len(close)
    chop = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], 
                     abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j],
                     abs(low[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        hl_range = hh - ll
        
        if hl_range > 1e-10:
            chop[i] = 100 * (np.log(atr_sum) / np.log(hl_range * period))
    
    return chop

def calculate_camarilla_levels(high, low, close):
    """Calculate Camarilla pivot levels - R3, R4, S3, S4"""
    n = len(high)
    r3 = np.full(n, np.nan, dtype=np.float64)
    r4 = np.full(n, np.nan, dtype=np.float64)
    s3 = np.full(n, np.nan, dtype=np.float64)
    s4 = np.full(n, np.nan, dtype=np.float64)
    
    for i in range(1, n):
        h = high[i]
        l = low[i]
        c = close[i]
        rng = h - l
        
        # Standard Camarilla levels
        r4[i] = c + rng * 1.1 / 2 + rng * 0.55
        r3[i] = c + rng * 1.1 / 4 + rng * 0.275
        s3[i] = c - rng * 1.1 / 4 - rng * 0.275
        s4[i] = c - rng * 1.1 / 2 - rng * 0.55
    
    return r3, r4, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Camarilla levels ===
    r3_1d, r4_1d, s3_1d, s4_1d = calculate_camarilla_levels(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align to 4h
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === Calculate local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    
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
    
    warmup = 60  # Need enough bars for ATR-14 + CHOP-14 + volume MA
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === REGIME CHECK (Choppiness) ===
        chop_val = chop[i]
        is_trending = chop_val < 50.0  # Trending - follow breakouts
        is_ranging = chop_val > 61.8   # Ranging - fade to extremes
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA LEVELS (from 1d) ===
        r3 = r3_aligned[i]
        r4 = r4_aligned[i]
        s3 = s3_aligned[i]
        s4 = s4_aligned[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            if is_trending and vol_spike:
                # TRENDING REGIME: Breakout trading
                # Long if price breaks above R3 with volume
                if close[i] > r3 and close[i-1] <= r3:
                    desired_signal = SIZE
                
                # Short if price breaks below S3 with volume
                if close[i] < s3 and close[i-1] >= s3:
                    desired_signal = -SIZE
            
            elif is_ranging:
                # RANGING REGIME: Fade to extremes
                # Long if price approaches S4 (oversold)
                if close[i] < s4:
                    desired_signal = SIZE
                
                # Short if price approaches R4 (overbought)
                if close[i] > r4:
                    desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.5 ATR trailing) ===
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
        
        # === EXIT: Target levels or regime change ===
        exit_triggered = False
        
        if in_position and position_side > 0:
            # Long exit: take profit at R4 or stoploss
            if close[i] >= r4:
                exit_triggered = True
        
        if in_position and position_side < 0:
            # Short exit: take profit at S4 or stoploss
            if close[i] <= s4:
                exit_triggered = True
        
        if exit_triggered:
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