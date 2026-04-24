#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme Reversal with 1d EMA50 Trend Filter and Volume Spike.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when Williams %R(14) crosses above -80 from below AND price > 1d EMA50 AND volume > 2.0 * 4h volume MA(20);
         Short when Williams %R(14) crosses below -20 from above AND price < 1d EMA50 AND volume > 2.0 * 4h volume MA(20).
- Exit: Long exits when Williams %R(14) crosses below -20 from above; Short exits when Williams %R(14) crosses above -80 from below.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying oversold bounces in uptrend) and bear (selling overbought rallies in downtrend) with volume confirmation to avoid false signals.
- Williams %R identifies reversal points; EMA50 ensures trend alignment; volume spike confirms participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA50 for 1d
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams %R(14) on 4h data
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    denom = highest_high - lowest_low
    williams_r = np.where(denom != 0, -100 * (highest_high - close) / denom, -50.0)
    
    # Get 4h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)  # EMA50 needs 50, volume MA needs 20, Williams %R needs 14
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_wr = williams_r[i]
        curr_wr_prev = williams_r[i-1]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R crosses above -80 from below AND price > 1d EMA50 (uptrend)
                if curr_wr > -80.0 and curr_wr_prev <= -80.0 and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R crosses below -20 from above AND price < 1d EMA50 (downtrend)
                elif curr_wr < -20.0 and curr_wr_prev >= -20.0 and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Williams %R crosses below -20 from above
            if curr_wr < -20.0 and curr_wr_prev >= -20.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R crosses above -80 from below
            if curr_wr > -80.0 and curr_wr_prev <= -80.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0