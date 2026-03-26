#!/usr/bin/env python3
"""
Experiment #011: 6h Camarilla Pivots + 1d Trend + Volume Confirmation

HYPOTHESIS: Camarilla pivot levels (S3/S4/R3/R4) derived from 1d data create 
natural support/resistance that price respects across all market conditions.
- S3/R3: Mean reversion levels (price fades back to "fair value")
- S4/R4: Breakout levels (continuation signal when broken)

Works in BOTH bull AND bear:
- BULL: Price bounces at S3, breaks through R3 with volume → long continuation
- BEAR: Price rejects at R3, breaks through S3 with volume → short continuation  
- RANGE: Price oscillates between S3/R3 (mean reversion)

WHY 6h: Balances fee drag (vs 4h) with opportunity (vs 12h). 6h bars give 
Camarilla levels time to "set up" while capturing daily volatility.

KEY INSIGHT from DB winners: tight entries + volume confirm + price structure.
Camarilla IS the price structure. Simple 2-3 conditions max.

TARGET: 75-200 total trades over 4 years. HARD MAX: 300.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_1d_trend_vol_v1"
timeframe = "6h"
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

def calculate_camarilla_levels_1d(high, low, close, prev_close):
    """
    Calculate Camarilla pivot levels from daily data.
    Using classic Camarilla equations:
    - S1 = prev_close - (prev_high - prev_low) * 1.1/12
    - S2 = prev_close - (prev_high - prev_low) * 1.1/6
    - S3 = prev_close - (prev_high - prev_low) * 1.1/4
    - S4 = prev_close - (prev_high - prev_low) * 1.1/2
    - R1 = prev_close + (prev_high - prev_low) * 1.1/12
    - R2 = prev_close + (prev_high - prev_low) * 1.1/6
    - R3 = prev_close + (prev_high - prev_low) * 1.1/4
    - R4 = prev_close + (prev_high - prev_low) * 1.1/2
    """
    prev_range = high - low
    
    s1 = prev_close - prev_range * (1.1 / 12)
    s2 = prev_close - prev_range * (1.1 / 6)
    s3 = prev_close - prev_range * (1.1 / 4)
    s4 = prev_close - prev_range * (1.1 / 2)
    
    r1 = prev_close + prev_range * (1.1 / 12)
    r2 = prev_close + prev_range * (1.1 / 6)
    r3 = prev_close + prev_range * (1.1 / 4)
    r4 = prev_close + prev_range * (1.1 / 2)
    
    return s1, s2, s3, s4, r1, r2, r3, r4

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA200 for trend direction
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma_200_aligned = align_htf_to_ltf(prices, df_1d, sma_200_1d)
    
    # Local indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Volume for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Calculate 1d Camarilla levels - we need previous day's H/L/C for each 1d bar
    # S3/R3 are the main levels for mean reversion
    # S4/R4 are breakout levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily Camarilla levels
    n_1d = len(daily_close)
    s3_daily = np.full(n_1d, np.nan)
    r3_daily = np.full(n_1d, np.nan)
    s4_daily = np.full(n_1d, np.nan)
    r4_daily = np.full(n_1d, np.nan)
    
    for i in range(1, n_1d):  # Start from 1 (need prev day)
        prev_h = daily_high[i-1]
        prev_l = daily_low[i-1]
        prev_c = daily_close[i-1]
        prev_range = prev_h - prev_l
        
        s3_daily[i] = prev_c - prev_range * (1.1 / 4)
        r3_daily[i] = prev_c + prev_range * (1.1 / 4)
        s4_daily[i] = prev_c - prev_range * (1.1 / 2)
        r4_daily[i] = prev_c + prev_range * (1.1 / 2)
    
    # Align to 6h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_daily)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_daily)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_daily)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_daily)
    
    signals = np.zeros(n)
    SIZE = 0.25  # Conservative sizing
    
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
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(sma_200_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Get Camarilla levels for this bar
        s3 = s3_aligned[i]
        r3 = r3_aligned[i]
        s4 = s4_aligned[i]
        r4 = r4_aligned[i]
        
        if np.isnan(s3) or np.isnan(r3):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA200) ===
        price_above_1d_sma = close[i] > sma_200_aligned[i]
        trend_bull = price_above_1d_sma
        trend_bear = not price_above_1d_sma
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.3  # 30% above average
        
        # === DISTANCE FROM CAMARILLA LEVELS ===
        # How far is price from key levels?
        if not np.isnan(s4) and not np.isnan(r4):
            # Price position within daily range
            range_width = r4 - s4
            if range_width > 0:
                price_position = (close[i] - s4) / range_width  # 0 = at S4, 1 = at R4
            else:
                price_position = 0.5
        else:
            price_position = 0.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG ENTRY: Mean Reversion at S3 ===
            # Price dropped to S3 zone, bouncing back
            # Condition: close near S3, above S3, volume spike, in uptrend
            if trend_bull:
                # Calculate distance to S3
                dist_to_s3 = close[i] - s3
                s3_range = r3 - s3
                
                if s3_range > 0:
                    # Price within lower 30% of range (near S3)
                    if dist_to_s3 < s3_range * 0.3 and dist_to_s3 > 0:
                        if vol_spike:
                            desired_signal = SIZE
                    # Or price bounced from below S3
                    elif low[i] < s3 and close[i] > s3:
                        if vol_spike:
                            desired_signal = SIZE
            
            # === SHORT ENTRY: Mean Reversion at R3 ===
            # Price rallied to R3 zone, reversing down
            # Condition: close near R3, below R3, volume spike, in downtrend
            if trend_bear:
                dist_to_r3 = r3 - close[i]
                r3_range = r3 - s3
                
                if r3_range > 0:
                    # Price within upper 30% of range (near R3)
                    if dist_to_r3 < r3_range * 0.3 and dist_to_r3 > 0:
                        if vol_spike:
                            desired_signal = -SIZE
                    # Or price rejected at R3
                    elif high[i] > r3 and close[i] < r3:
                        if vol_spike:
                            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2.0 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.0 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.0 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === TIME-BASED EXIT (hold at least 4 bars = 1 day) ===
        bars_held = i - entry_bar
        
        if in_position and bars_held >= 4:
            # Exit on profit target or Camarilla reversal
            if position_side > 0:
                # Take profit at R3 or if rejected at R3
                if close[i] >= r3 - atr_14[i]:
                    desired_signal = 0.0
                elif high[i] > r3 and close[i] < r3:
                    desired_signal = 0.0  # Rejected at R3
            
            if position_side < 0:
                # Take profit at S3 or if bounced at S3
                if close[i] <= s3 + atr_14[i]:
                    desired_signal = 0.0
                elif low[i] < s3 and close[i] > s3:
                    desired_signal = 0.0  # Bounced at S3
        
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
                entry_bar = i
                if position_side > 0:
                    stop_price = entry_price - 2.0 * entry_atr
                else:
                    stop_price = entry_price + 2.0 * entry_atr
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