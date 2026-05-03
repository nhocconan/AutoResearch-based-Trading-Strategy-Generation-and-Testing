#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h EMA50 trend filter and volume spike
# Long when price breaks above Camarilla R3 level with 12h EMA50 uptrend and volume > 2.0x 20-bar average
# Short when price breaks below Camarilla S3 level with 12h EMA50 downtrend and volume > 2.0x 20-bar average
# Exit via ATR(14) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses Camarilla pivot levels from 4h timeframe for structure, 12h EMA50 for trend filter, volume for confirmation
# ATR multiplier 2.5 avoids premature exits in volatile markets. Discrete sizing 0.30 balances return and fee drag.
# Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_Camarilla_R3S3_12hEMA50_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Camarilla pivot levels (using prior 4h bar's OHLC)
    ph = prices['high'].shift(1).values  # prior 4h high
    pl = prices['low'].shift(1).values   # prior 4h low
    pc = prices['close'].shift(1).values # prior 4h close
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_r3 = pc + (ph - pl) * 1.1 / 4
    camarilla_s3 = pc - (ph - pl) * 1.1 / 4
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for EMA50 and ATR calculations)
    start_idx = 60  # EMA50 needs 50 bars + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 with 12h EMA50 uptrend and volume spike
            if close[i] > camarilla_r3[i] and ema_50_aligned[i] > ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = 0.30
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: price breaks below Camarilla S3 with 12h EMA50 downtrend and volume spike
            elif close[i] < camarilla_s3[i] and ema_50_aligned[i] < ema_50_aligned[i-1] and volume_spike[i]:
                signals[i] = -0.30
                position = -1
                entry_bar = i
                lowest_since_entry = low[i]
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            highest_since_entry = max(highest_since_entry, high[i])
            # ATR trailing stop: exit when price < highest_high_since_entry - 2.5 * ATR
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals