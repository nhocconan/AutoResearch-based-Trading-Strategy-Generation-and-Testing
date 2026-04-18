#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_R1_S1_Breakout_Volume_Filter_v1
Strategy: Camarilla R1/S1 breakout on 1h with 4h trend filter and 1d volume confirmation.
- 4h EMA34 trend: price > EMA34 = uptrend, price < EMA34 = downtrend
- 1d volume: current volume > 1.5x 20-period volume average
- 1h Camarilla: long on break above R1, short on break below S1
- Session filter: 08:00-20:00 UTC
- Position size: 0.20 (discrete to minimize fee churn)
Target: 15-30 trades/year (60-120 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA34 for trend
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d 20-period volume average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Pre-calculate 1h Camarilla levels from previous bar
    # Typical Price = (H + L + C) / 3
    typical_price = (high + low + close) / 3
    range_hl = high - low
    
    # Camarilla levels
    R1 = close + (1.1/12) * range_hl
    S1 = close - (1.1/12) * range_hl
    
    # Shift to get previous bar's levels (no look-ahead)
    R1_prev = np.roll(R1, 1)
    S1_prev = np.roll(S1, 1)
    R1_prev[0] = np.nan
    S1_prev[0] = np.nan
    
    # Pre-calculate session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(R1_prev[i]) or np.isnan(S1_prev[i])):
            signals[i] = 0.0
            continue
        
        # Trend condition from 4h EMA34
        uptrend = close[i] > ema_34_4h_aligned[i]
        downtrend = close[i] < ema_34_4h_aligned[i]
        
        # Volume confirmation from 1d
        vol_confirm = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + break above R1
            if uptrend and vol_confirm and high[i] > R1_prev[i]:
                signals[i] = 0.20
                position = 1
            # Short: downtrend + volume + break below S1
            elif downtrend and vol_confirm and low[i] < S1_prev[i]:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: trend change or break below S1 (reversal)
            if not uptrend or low[i] < S1_prev[i]:
                signals[i] = -0.20  # reverse to short
                position = -1
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: trend change or break above R1 (reversal)
            if not downtrend or high[i] > R1_prev[i]:
                signals[i] = 0.20  # reverse to long
                position = 1
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_R1_S1_Breakout_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0