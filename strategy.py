#!/usr/bin/env python3
"""
Experiment #7775: 6-hour weekly pivot range with volume confirmation and ATR-based risk management.
Hypothesis: Weekly pivot ranges (from Sunday candle) act as strong support/resistance on 6h timeframe. 
Price bouncing off S1/R1 with volume >1.5x 20-period MA and weekly trend filter (price > weekly EMA50) captures 
reversals in ranging markets and continuations in trending markets. Works in both bull and bear by fading extremes 
in range and following breakouts in trends. Targets 60-120 trades over 4 years.
"""

from mtf_data import get_htf_data, align_ltf_to_htf
import numpy as np
import pandas as pd

name = "exp_7775_6h_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
PIVOT_LOOKBACK = 5  # bars to confirm pivot hold
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_pivot(high, low, close):
    """Calculate standard pivot points: P = (H+L+C)/3, S1 = 2P-H, R1 = 2P-L"""
    p = (high + low + close) / 3.0
    s1 = 2 * p - high
    r1 = 2 * p - low
    return p, s1, r1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points from previous week
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate pivots for each weekly bar
    weekly_p = np.full_like(weekly_close, np.nan)
    weekly_s1 = np.full_like(weekly_close, np.nan)
    weekly_r1 = np.full_like(weekly_close, np.nan)
    
    for i in range(1, len(weekly_close)):  # Start from 1 to use previous week
        p, s1, r1 = calculate_pivot(weekly_high[i-1], weekly_low[i-1], weekly_close[i-1])
        weekly_p[i] = p
        weekly_s1[i] = s1
        weekly_r1[i] = r1
    
    # Align weekly pivots to 6h timeframe
    weekly_p_aligned = align_ltf_to_htf(prices, df_weekly, weekly_p)
    weekly_s1_aligned = align_ltf_to_htf(prices, df_weekly, weekly_s1)
    weekly_r1_aligned = align_ltf_to_htf(prices, df_weekly, weekly_r1)
    
    # Calculate weekly EMA for trend filter
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_aligned = align_ltf_to_htf(prices, df_weekly, weekly_ema)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD, PIVOT_LOOKBACK) + 1
    
    for i in range(start, n):
        # Skip if weekly data not available
        if np.isnan(weekly_p_aligned[i]) or np.isnan(weekly_ema_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend
        weekly_uptrend = close[i] > weekly_ema_aligned[i]
        weekly_downtrend = close[i] < weekly_ema_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Price near weekly S1/R1 (within 0.5% tolerance)
        near_s1 = np.abs((close[i] - weekly_s1_aligned[i]) / weekly_s1_aligned[i]) < 0.005 if weekly_s1_aligned[i] != 0 else False
        near_r1 = np.abs((close[i] - weekly_r1_aligned[i]) / weekly_r1_aligned[i]) < 0.005 if weekly_r1_aligned[i] != 0 else False
        
        # Entry logic: 
        # In weekly uptrend: buy near S1 (support), sell near R1 (resistance) 
        # In weekly downtrend: sell near R1, buy near S1
        long_entry = False
        short_entry = False
        
        if weekly_uptrend:
            # In uptrend, buy dips to S1, sell rallies to R1
            long_entry = near_s1 and volume_confirmed
            # Short on R1 rejection in uptrend (less common)
            short_entry = near_r1 and volume_confirmed and close[i] < weekly_r1_aligned[i]  # rejection
        else:  # weekly_downtrend
            # In downtrend, sell rallies to R1, buy dips to S1
            short_entry = near_r1 and volume_confirmed
            long_entry = near_s1 and volume_confirmed and close[i] > weekly_s1_aligned[i]  # bounce
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals

def align_ltf_to_htf(ltf_df, htf_df, htf_values):
    """Helper function to align HTF values to LTF using forward fill"""
    # Create a DatetimeIndex for LTF
    ltf_index = ltf_df.index if isinstance(ltf_df.index, pd.DatetimeIndex) else pd.to_datetime(ltf_df['open_time'])
    # Create a DatetimeIndex for HTF
    htf_index = htf_df.index if isinstance(htf_df.index, pd.DatetimeIndex) else pd.to_datetime(htf_df['open_time'])
    
    # Create series with HTF values indexed by HTF timestamps
    htf_series = pd.Series(htf_values, index=htf_index)
    # Forward fill to LTF index
    aligned_series = htf_series.reindex(ltf_index, method='ffill')
    return aligned_series.values

# Override the import to use our local function
from mtf_data import get_htf_data
align_htf_to_ltf = align_ltf_to_htf  # rename for compatibility