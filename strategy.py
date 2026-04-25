#!/usr/bin/env python3
"""
12h Camarilla R3/S3 Breakout with 1d EMA34 Trend and Volume Spike
Hypothesis: Camarilla pivot levels (R3/S3) act as strong support/resistance on 1d timeframe.
Breakout above R3 or below S3 with 1d EMA34 trend alignment and volume spike captures
institutional breakout moves. Works in both bull (breakouts continuation) and bear
(breakdowns acceleration) markets. Targets 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Camarilla pivots and EMA34 trend (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R3, S3) from previous 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4)
    #          S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # where C = (H+L+Close)/3 (typical price)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    hl_range = df_1d['high'] - df_1d['low']
    
    camarilla_r3 = typical_price + (hl_range * 1.1 / 4)
    camarilla_s3 = typical_price - (hl_range * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe (1-bar delay for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # Calculate 34-period EMA on 1d close for trend
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume MA for 12h volume confirmation
    vol_ma_20_12h = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20_12h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA34 and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma_20_12h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_ma_12h = vol_ma_20_12h[i]
        
        # Volume confirmation: current 12h volume > 2.5 * 20-period average
        volume_confirm = curr_volume > 2.5 * vol_ma_12h
        
        if position == 0:
            # Look for breakout signals
            # Long: Close breaks above R3 AND price > EMA34 (uptrend) AND volume confirmation
            long_breakout = (curr_close > r3_level) and (curr_close > ema_trend) and volume_confirm
            # Short: Close breaks below S3 AND price < EMA34 (downtrend) AND volume confirmation
            short_breakout = (curr_close < s3_level) and (curr_close < ema_trend) and volume_confirm
            
            if long_breakout:
                signals[i] = 0.30
                position = 1
            elif short_breakout:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Close falls below EMA34 OR Camarilla S3 retest (mean reversion)
            if (curr_close < ema_trend) or (curr_close < s3_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short position management
            # Exit: Close rises above EMA34 OR Camarilla R3 retest (mean reversion)
            if (curr_close > ema_trend) or (curr_close > r3_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0