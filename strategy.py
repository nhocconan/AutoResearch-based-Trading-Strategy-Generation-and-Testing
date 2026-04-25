#!/usr/bin/env python3
"""
1h EMA Trend Filter + 4h RSI Mean Reversion + Volume Spike
Hypothesis: On 1h timeframe, use 4h EMA for trend direction (filter) and 4h RSI for mean reversion entries.
Only take trades aligned with higher timeframe trend: long when 4h EMA rising, short when falling.
Enter on 1h RSI extreme (oversold/overbought) with volume spike confirmation.
Uses session filter (08-20 UTC) to avoid low-liquidity periods.
Target: 60-120 trades over 4 years (15-30/year) to minimize fee drag.
Works in bull/bear: trend filter avoids counter-trend trades, mean reversion catches pullbacks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF indicators (called ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA(20) for trend direction
    ema_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 4h RSI(14) for mean reversion signals
    delta = pd.Series(df_4h['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = rsi_4h.fillna(50).values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1h volume average for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for 4h EMA (20) + 4h RSI (14) + 1h VolMA (20)
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_4h_aligned[i]
        rsi_value = rsi_4h_aligned[i]
        vol_average = vol_ma_20[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        volume_spike = curr_volume > 2.0 * vol_average
        
        # Mean reversion conditions
        oversold = rsi_value < 30
        overbought = rsi_value > 70
        
        # Trend filter: only trade in direction of 4h EMA slope
        # Use EMA slope approximation: current EMA vs EMA 3 periods ago
        if i >= 3:
            ema_slope = ema_trend - ema_4h_aligned[i-3]
            trend_up = ema_slope > 0
            trend_down = ema_slope < 0
        else:
            trend_up = False
            trend_down = False
        
        # Exit conditions: RSI mean reversion or volume drying up
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit long when RSI returns to neutral (50) or volume drops
                if rsi_value >= 50 or not volume_spike:
                    exit_signal = True
                    
            elif position == -1:
                # Exit short when RSI returns to neutral (50) or volume drops
                if rsi_value <= 50 or not volume_spike:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions: RSI extreme + volume spike + trend filter
        if position == 0:
            # Long: oversold RSI + volume spike + uptrend on 4h
            long_condition = oversold and volume_spike and trend_up
            # Short: overbought RSI + volume spike + downtrend on 4h
            short_condition = overbought and volume_spike and trend_down
            
            if long_condition:
                signals[i] = 0.20
                position = 1
            elif short_condition:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            signals[i] = 0.20
        elif position == -1:
            signals[i] = -0.20
    
    return signals

name = "1h_EMA20_Trend_RSI14_MeanRev_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0