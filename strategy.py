#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
# Long when: Williams %R(14) crosses above -80 (oversold reversal) AND close > 1d EMA34 AND volume > 2.0x 24-bar average
# Short when: Williams %R(14) crosses below -20 (overbought reversal) AND close < 1d EMA34 AND volume > 2.0x 24-bar average
# Exit via ATR(24) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 4h Williams %R for mean-reversion edge in ranging markets, 1d EMA34 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.28 balances return and fee drag. Target: 75-200 total trades over 4 years = 19-50/year.

name = "4h_WilliamsR_1dEMA34_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Williams %R(14)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    williams_r = np.full(len(close_4h), np.nan)
    for i in range(14, len(close_4h)):
        highest_high = np.max(high_4h[i-14:i])
        lowest_low = np.min(low_4h[i-14:i])
        if highest_high != lowest_low:
            williams_r[i] = (highest_high - close_4h[i]) / (highest_high - lowest_low) * -100
        else:
            williams_r[i] = -50  # neutral when range is zero
    
    # Align 4h Williams %R to 4h timeframe (completed 4h bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    
    # Calculate 1d EMA34 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 4h ATR(24) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=24, adjust=False, min_periods=24).mean().values
    
    # Volume confirmation (2.0x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for Williams %R, EMA, ATR calculations)
    start_idx = 50 + 34 + 24 + 5  # EMA34(1d) warmup + Williams %R(14) + ATR(24) + buffer
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R crosses above -80 (oversold reversal) with volume spike AND bullish trend (close > 1d EMA34)
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                volume_spike[i] and close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.28
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: Williams %R crosses below -20 (overbought reversal) with volume spike AND bearish trend (close < 1d EMA34)
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  volume_spike[i] and close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.28
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
                signals[i] = 0.28
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_since_entry = min(lowest_since_entry, low[i])
            # ATR trailing stop: exit when price > lowest_low_since_entry + 2.5 * ATR
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals