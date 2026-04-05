#!/usr/bin/env python3
"""
exp_6999_6h_camarilla1d_pivot_v3
Hypothesis: 6h Camarilla pivot reversals with 1d trend filter. 
In uptrend (price > 1d EMA50): long at S1/S2, short at R3/R4. 
In downtrend (price < 1d EMA50): short at R1/R2, long at S3/S4. 
Volume confirmation filters false breaks. Uses discrete sizing (0.25) to minimize fees.
Targets 50-150 trades over 4 years by requiring confluence of pivot level, trend, and volume.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6999_6h_camarilla1d_pivot_v3"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_LOOKBACK = 20  # 1d lookback for pivot calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 20  # ~5 days (6h bars)
EMA_PERIOD = 50

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for EMA and pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1d Camarilla pivots (based on previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].shift(1).values  # Previous day close
    
    # True range for volatility
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First value
    
    # Camarilla levels: based on previous day's close and range
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.25 * (High - Low)
    # R2 = Close + 1.166 * (High - Low)
    # R1 = Close + 1.083 * (High - Low)
    # PP = (High + Low + Close) / 3
    # S1 = Close - 1.083 * (High - Low)
    # S2 = Close - 1.166 * (High - Low)
    # S3 = Close - 1.25 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    
    hl_range = high_1d - low_1d
    r4_1d = close_1d_prev + 1.5 * hl_range
    r3_1d = close_1d_prev + 1.25 * hl_range
    r2_1d = close_1d_prev + 1.166 * hl_range
    r1_1d = close_1d_prev + 1.083 * hl_range
    pp_1d = (high_1d + low_1d + close_1d_prev) / 3.0
    s1_1d = close_1d_prev - 1.083 * hl_range
    s2_1d = close_1d_prev - 1.166 * hl_range
    s3_1d = close_1d_prev - 1.25 * hl_range
    s4_1d = close_1d_prev - 1.5 * hl_range
    
    # Align all pivot levels to 6h timeframe
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(CAMARILLA_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD, EMA_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine trend direction from 1d EMA50
        daily_uptrend = close[i] > ema_1d_aligned[i]
        daily_downtrend = close[i] < ema_1d_aligned[i]
        
        # Camarilla pivot logic with trend alignment
        # In uptrend: buy pullbacks to support (S1/S2), sell rallies to resistance (R3/R4)
        # In downtrend: sell rallies to resistance (R1/R2), buy pullbacks to support (S3/S4)
        long_signal = False
        short_signal = False
        
        if daily_uptrend:
            # Long near support in uptrend
            if (close[i] <= s1_1d_aligned[i] * 1.002 or  # Allow small buffer
                close[i] <= s2_1d_aligned[i] * 1.002) and vol_confirmed:
                long_signal = True
            # Short at resistance in uptrend (fade strength)
            if (close[i] >= r3_1d_aligned[i] * 0.998 or  # Allow small buffer
                close[i] >= r4_1d_aligned[i] * 0.998) and vol_confirmed:
                short_signal = True
        elif daily_downtrend:
            # Short near resistance in downtrend
            if (close[i] >= r1_1d_aligned[i] * 0.998 or  # Allow small buffer
                close[i] >= r2_1d_aligned[i] * 0.998) and vol_confirmed:
                short_signal = True
            # Long at support in downtrend (fade weakness)
            if (close[i] <= s3_1d_aligned[i] * 1.002 or  # Allow small buffer
                close[i] <= s4_1d_aligned[i] * 1.002) and vol_confirmed:
                long_signal = True
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals