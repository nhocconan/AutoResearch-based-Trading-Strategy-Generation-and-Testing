#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme + 1d EMA50 Trend Filter + Volume Spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d EMA50 for trend filter (price > EMA50 = uptrend, price < EMA50 = downtrend).
- Entry: Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 2.0 * 6h volume MA(20);
         Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 2.0 * 6h volume MA(20).
- Exit: Long exits when Williams %R > -20; Short exits when Williams %R < -80.
- Signal size: 0.25 discrete to balance capture and fee control.
- Works in bull (buying oversold in uptrend) and bear (selling overbought in downtrend) with volume confirmation to avoid false signals.
- Williams %R identifies exhaustion points; trend filter ensures alignment with higher timeframe direction.
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
    
    # Align EMA50 to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Williams %R on 6h data (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Get 6h data for volume MA(20)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, period, 20)  # EMA50 needs 50, Williams %R needs 14, volume MA needs 20
    
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
        curr_williams_r = williams_r[i]
        
        # Volume confirmation: 2.0x threshold
        vol_confirm = curr_volume > 2.0 * vol_ma[i]
        
        if position == 0:
            # Check for entry signals
            if vol_confirm:
                # Long: Williams %R < -80 (oversold) AND price > 1d EMA50 (uptrend)
                if curr_williams_r < -80 and curr_close > ema_50_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: Williams %R > -20 (overbought) AND price < 1d EMA50 (downtrend)
                elif curr_williams_r > -20 and curr_close < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when Williams %R > -20 (exit oversold)
            if curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when Williams %R < -80 (exit overbought)
            if curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0