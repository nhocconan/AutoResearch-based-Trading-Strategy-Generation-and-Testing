#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d ATR(14) for volatility measurement
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align HTF indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Hour filter: 8-20 UTC (only trade during active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 14  # Wait for ATR warmup
    
    for i in range(start_idx, n):
        # Skip if ATR data is not ready
        if np.isnan(atr_14_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility contraction filter: look for low volatility periods
        # ATR contraction: current ATR < 0.7 * 20-period average ATR
        if i >= 34:  # Need 14 + 20 for ATR MA
            atr_ma = np.nanmean(atr_14_aligned[i-20:i]) if not np.isnan(np.nanmean(atr_14_aligned[i-20:i])) else atr_14_aligned[i]
            vol_contract = atr_14_aligned[i] < 0.7 * atr_ma
        else:
            vol_contract = False
        
        # Price action: look for breakouts from recent range
        # 20-period high/low for breakout detection
        if i >= 20:
            period_high = np.nanmax(high[i-20:i])
            period_low = np.nanmin(low[i-20:i])
            
            # Breakout conditions with volume confirmation
            vol_ma = np.nanmean(volume[i-5:i]) if i >= 5 else volume[i]
            vol_surge = volume[i] > 1.5 * vol_ma
            
            # Long breakout: price breaks above recent range with volume
            long_breakout = (close[i] > period_high) and vol_surge and vol_contract
            
            # Short breakout: price breaks below recent range with volume
            short_breakout = (close[i] < period_low) and vol_surge and vol_contract
        else:
            long_breakout = False
            short_breakout = False
        
        # Exit conditions: volatility expansion or opposing breakout
        vol_expand = atr_14_aligned[i] > 1.5 * atr_ma if i >= 34 else False
        opposing_breakout = short_breakout if position == 1 else long_breakout if position == -1 else False
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (vol_expand or opposing_breakout) and position != 0:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_VolContraction_Breakout"
timeframe = "4h"
leverage = 1.0