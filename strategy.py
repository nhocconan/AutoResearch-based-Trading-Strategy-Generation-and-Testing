#!/usr/bin/env python3
"""
Experiment #011: 6h Camarilla Pivot + Williams %R + Choppiness Regime

HYPOTHESIS: Camarilla pivots provide key S3/R3 reversal zones not captured by
Donchian. Combined with Williams %R momentum (not RSI, not ADX) and Choppiness
regime filter, this targets mean-reversion at pivot levels in trending markets.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: R3 breakout + %R>80 + CHOP<50 = momentum continuation long
- Bear: S3 breakdown + %R<20 + CHOP<50 = momentum continuation short
- Range: Fade R3/S3 when CHOP>61.8, exit when CHOP<38.2 flips to trend

NOVELTY vs failed attempts:
- Uses Camarilla (not Donchian) — different price structure
- Uses Williams %R (not RSI/ADX) — more sensitive momentum
- Combines pivot fade + breakout logic with regime filter

EXPECTED TRADES: 80-150 total over 4 years (20-37/year per symbol)
- R3/S3 touches happen ~2-4x per week per symbol
- %R filter reduces by ~40%
- Choppiness filter reduces by ~50%
- Final: ~80-150 total (within target)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_wr_chop_v1"
timeframe = "6h"
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

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    wr = np.full(n, np.nan)
    for i in range(period - 1, n):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        if highest_high != lowest_low:
            wr[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
    return wr

def calculate_choppiness_index(high, low, close, period=14):
    """Choppiness Index - measures trend vs range"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    for i in range(period - 1, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        
        range_sum = hh - ll
        
        if range_sum > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_daily_camarilla(high_d, low_d, close_d):
    """Daily Camarilla pivot levels"""
    h_l = high_d - low_d
    h_c = high_d - close_d
    c_l = close_d - low_d
    
    r4 = close_d + h_l * 1.1 / 2
    r3 = close_d + h_l * 1.1 / 4
    r4_alt = h_c + close_d + h_l / 2  # Alternative calculation
    r3_alt = h_c + close_d + h_l / 4
    s3 = close_d - h_l * 1.1 / 4
    s4 = close_d - h_l * 1.1 / 2
    
    return r4, r3, s3, s4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA50 for trend alignment (align to 6h)
    daily_ema50 = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Daily Camarilla levels from previous day (align to 6h)
    df_1d_shifted = df_1d.copy()
    df_1d_shifted['r4'] = df_1d['high'].shift(1) + 2 * (df_1d['close'].shift(1) - df_1d['low'].shift(1))
    df_1d_shifted['r3'] = df_1d['close'].shift(1) + (df_1d['high'].shift(1) - df_1d['low'].shift(1)) / 2
    df_1d_shifted['s3'] = df_1d['close'].shift(1) - (df_1d['high'].shift(1) - df_1d['low'].shift(1)) / 2
    df_1d_shifted['s4'] = df_1d['close'].shift(1) - 2 * (df_1d['high'].shift(1) - df_1d['close'].shift(1))
    
    r3_aligned = align_htf_to_ltf(prices, df_1d_shifted, df_1d_shifted['r3'].values)
    s3_aligned = align_htf_to_ltf(prices, df_1d_shifted, df_1d_shifted['s3'].values)
    r4_aligned = align_htf_to_ltf(prices, df_1d_shifted, df_1d_shifted['r4'].values)
    s4_aligned = align_htf_to_ltf(prices, df_1d_shifted, df_1d_shifted['s4'].values)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    williams_r = calculate_williams_r(high, low, close, period=14)
    choppiness = calculate_choppiness_index(high, low, close, period=14)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    stop_price = 0.0
    target_price = 0.0
    
    warmup = 100  # Enough for all indicators
    
    for i in range(warmup, n):
        # NaN checks
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(williams_r[i]) or np.isnan(choppiness[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema50_aligned[i]) or np.isnan(r3_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === CHOPPINESS REGIME ===
        # CHOP < 38.2 = trending (follow momentum)
        # CHOP > 61.8 = ranging (fade pivots)
        chop_trending = choppiness[i] < 38.2
        chop_ranging = choppiness[i] > 61.8
        
        # === WILLIAMS %R MOMENTUM ===
        # %R > -20 = overbought (bull momentum)
        # %R < -80 = oversold (bear momentum)
        bull_momentum = williams_r[i] > -20
        bear_momentum = williams_r[i] < -80
        
        # === TREND DIRECTION: Daily EMA50 ===
        bull_trend = close[i] > ema50_aligned[i]
        bear_trend = close[i] < ema50_aligned[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === CAMARILLA PIVOT SIGNALS ===
        # R3 breakout: price crosses above yesterday's R3
        r3_breakout = close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i]
        # S3 breakdown: price crosses below yesterday's S3
        s3_breakdown = close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i]
        
        # === MINIMUM HOLD: 4 bars to reduce fee churn ===
        min_hold_passed = (i - entry_bar) >= 4 if in_position else True
        
        # === EXITS ===
        if in_position:
            # Update stop to trailing
            if position_side > 0:
                # Long: trail stop below entry - 1.5 ATR
                new_stop = high[i] - 2.0 * atr_14[i]
                stop_price = min(stop_price, new_stop) if stop_price > 0 else new_stop
                stop_hit = low[i] < stop_price
            else:
                # Short: trail stop above entry + 1.5 ATR
                new_stop = low[i] + 2.0 * atr_14[i]
                stop_price = max(stop_price, new_stop) if stop_price > 0 else new_stop
                stop_hit = high[i] > stop_price
            
            # Trend exit: price crosses EMA50 against position
            trend_exit = (position_side > 0 and close[i] < ema50_aligned[i]) or \
                        (position_side < 0 and close[i] > ema50_aligned[i])
            
            # Target hit: price reaches R4/S4 level
            target_hit = (position_side > 0 and high[i] >= r4_aligned[i]) or \
                        (position_side < 0 and low[i] <= s4_aligned[i])
            
            if stop_hit or (min_hold_passed and trend_exit) or target_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
            else:
                signals[i] = position_side * SIZE
        
        # === NEW POSITIONS ===
        if not in_position:
            # LONG: R3 breakout + bull momentum + bull trend + vol spike
            if r3_breakout and bull_momentum and bull_trend and vol_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                stop_price = entry_price - 2.5 * entry_atr
                signals[i] = SIZE
            
            # SHORT: S3 breakdown + bear momentum + bear trend + vol spike
            elif s3_breakdown and bear_momentum and bear_trend and vol_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                stop_price = entry_price + 2.5 * entry_atr
                signals[i] = -SIZE
            
            # RANGE FADE: If ranging market, fade R3/S3 with opposite logic
            elif chop_ranging and min_hold_passed:
                # Short at R3 in ranging (mean reversion)
                if high[i] > r3_aligned[i] and close[i] < r3_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    stop_price = entry_price + 2.0 * entry_atr
                    signals[i] = -SIZE
                
                # Long at S3 in ranging (mean reversion)
                elif low[i] < s3_aligned[i] and close[i] > s3_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    entry_atr = atr_14[i]
                    entry_bar = i
                    stop_price = entry_price - 2.0 * entry_atr
                    signals[i] = SIZE
    
    return signals