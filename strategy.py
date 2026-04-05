#!/usr/bin/env python3
"""
exp_7418_1d_donchian20_1w_ema_vol_v1
Hypothesis: Daily timeframe with weekly EMA trend filter. Daily Donchian(20) breakouts
with volume confirmation and weekly EMA trend filter. Designed for low trade frequency
(30-100 total over 4 years) to minimize fee drag. Works in bull/bear via weekly EMA:
only long when above weekly EMA, short when below. Uses weekly timeframe for trend
filter to reduce whipsaws and capture major trends.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7418_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_WEEKLY_PERIOD = 50
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_DAYS = 10

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using weekly for EMA trend
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA
    close_weekly = df_weekly['close'].values
    ema_weekly = pd.Series(close_weekly).ewm(span=EMA_WEEKLY_PERIOD, adjust=False, min_periods=EMA_WEEKLY_PERIOD).mean().values
    
    # Align to daily timeframe
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Calculate daily indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (daily)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation (daily)
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss (daily)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    days_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_WEEKLY_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        days_since_entry += 1
        
        # Skip if weekly data not available
        if np.isnan(ema_weekly_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                days_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                days_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and days_since_entry >= MAX_HOLD_DAYS:
            signals[i] = 0.0
            position = 0
            days_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on weekly EMA
        above_weekly_ema = close[i] > ema_weekly_aligned[i]
        below_weekly_ema = close[i] < ema_weekly_aligned[i]
        
        # Continuation breakouts in trending market
        continuation_long = above_weekly_ema and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = below_weekly_ema and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Breakout retest entries (pullback to breakout level with volume)
        retest_long = above_weekly_ema and (close[i] <= highest_high[i-1] * 1.005) and (close[i] >= lowest_low[i-1]) and vol_confirmed
        retest_short = below_weekly_ema and (close[i] >= lowest_low[i-1] * 0.995) and (close[i] <= highest_high[i-1]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if continuation_long or retest_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                days_since_entry = 0
            elif continuation_short or retest_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                days_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals