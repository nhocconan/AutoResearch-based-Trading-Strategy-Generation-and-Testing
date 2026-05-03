#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot (R3/S3) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when: price breaks above 4h Camarilla R3 AND close > 1d EMA34 AND volume > 1.5x 20-bar average
# Short when: price breaks below 4h Camarilla S3 AND close < 1d EMA34 AND volume > 1.5x 20-bar average
# Exit: close-based reversal (opposite Camarilla level break) or ATR(14) trailing stop.
# Uses 4h Camarilla for structure (reduces noise), 1d EMA34 for trend filter, volume for confirmation.
# Discrete sizing 0.20 balances return and fee drag. Target: 60-150 total trades over 4 years = 15-37/year.
# Session filter: 08-20 UTC to avoid low-volume Asian session noise.

name = "1h_Camarilla_R3S3_1dEMA34_Volume_ATRStop_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla pivot levels (R3, S3, Pivot)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Typical price for pivot calculation
    typical_4h = (high_4h + low_4h + close_4h) / 3.0
    # Camarilla levels based on previous day's range
    camarilla_pivot = pd.Series(typical_4h).shift(1).values  # Use previous 4h bar's typical price
    camarilla_range = (pd.Series(high_4h).shift(1) - pd.Series(low_4h).shift(1)).values
    
    # Camarilla R3 and S3 levels
    camarilla_r3 = camarilla_pivot + camarilla_range * 1.1 / 4.0
    camarilla_s3 = camarilla_pivot - camarilla_range * 1.1 / 4.0
    
    # Align 4h Camarilla levels to 1h timeframe (completed 4h bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1h ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Session filter: 08-20 UTC (precomputed for performance)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for ATR, EMA calculations)
    start_idx = 34 + 20 + 5  # EMA34 warmup + volume MA + buffer
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above 4h Camarilla R3 with volume spike AND bullish trend (close > 1d EMA34)
            if close[i] > camarilla_r3_aligned[i] and volume_spike[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.20
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below 4h Camarilla S3 with volume spike AND bearish trend (close < 1d EMA34)
            elif close[i] < camarilla_s3_aligned[i] and volume_spike[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.20
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.0 * ATR
            if close[i] < highest_since_entry - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.0 * ATR
            if close[i] > lowest_since_entry + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals