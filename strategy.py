#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Williams %R reversal + 4h EMA200 trend filter + volume spike + session filter (08-20 UTC)
# Long when Williams %R < -80 (oversold) AND price > 4h EMA200 (uptrend) AND volume > 2x 20-bar average AND session 08-20 UTC
# Short when Williams %R > -20 (overbought) AND price < 4h EMA200 (downtrend) AND volume > 2x 20-bar average AND session 08-20 UTC
# Exit via ATR trailing stop: long exit when price < highest_high_since_entry - 2.0 * ATR, short exit when price > lowest_low_since_entry + 2.0 * ATR
# Williams %R captures mean reversals in ranging markets, 4h EMA200 provides higher-timeframe bias, volume confirms conviction.
# Target: 60-150 total trades over 4 years = 15-37/year. Uses discrete sizing (0.20) to minimize fee drag.

name = "1h_WilliamsR_4hEMA200_Trend_Volume_Session"
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
    
    # Load 4h data ONCE before loop for EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h EMA200 for trend filter
    close_4h = df_4h['close'].values
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Williams %R (14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(14, 20, 200) + 1  # Williams %R(14) + volume MA(20) + EMA200(4h)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(williams_r[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Williams %R < -80 (oversold) AND price > 4h EMA200 (uptrend) AND volume spike AND session
            if (williams_r[i] < -80 and 
                close[i] > ema_200_aligned[i] and 
                volume_spike[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
                entry_bar = i
                highest_since_entry = high[i]
            # Short entry: Williams %R > -20 (overbought) AND price < 4h EMA200 (downtrend) AND volume spike AND session
            elif (williams_r[i] > -20 and 
                  close[i] < ema_200_aligned[i] and 
                  volume_spike[i] and 
                  session_filter[i]):
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