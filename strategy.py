#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d trend filter and volume confirmation to reduce noise.
# Uses 4h EMA200 for long-term trend direction and 1d VWAP deviation for mean reversion in trend.
# Entry: Long when price > 4h EMA200, price < 1d VWAP, and volume spike; Short when price < 4h EMA200, price > 1d VWAP, and volume spike.
# Exit: Opposite condition (price crosses EMA200 or reverts to VWAP).
# Uses strict conditions to limit trades (~15-25/year) and avoid overtrading.
# Session filter (08-20 UTC) reduces noise during low-activity periods.
name = "1h_EMA200_VWAP_MeanReversion"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA200 for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    ema_200_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # 1d VWAP for mean reversion reference
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate VWAP for each day: sum(price * volume) / sum(volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap = (typical_price * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap = vwap.values
    
    # Align VWAP to 1h timeframe (waits for daily close)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    # Session filter: 08-20 UTC (inclusive)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: uptrend (price > EMA200) but pulled back to VWAP (price < VWAP) with volume
            if (close[i] > ema_200_4h_aligned[i] and 
                close[i] < vwap_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short: downtrend (price < EMA200) but bounced to VWAP (price > VWAP) with volume
            elif (close[i] < ema_200_4h_aligned[i] and 
                  close[i] > vwap_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if trend breaks (price < EMA200) or mean reversion complete (price >= VWAP)
            if (close[i] < ema_200_4h_aligned[i]) or (close[i] >= vwap_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if trend breaks (price > EMA200) or mean reversion complete (price <= VWAP)
            if (close[i] > ema_200_4h_aligned[i]) or (close[i] <= vwap_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals