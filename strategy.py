#!/usr/bin/env python3
"""
1h Camarilla Pivot Breakout + 4h EMA50 Trend + Volume Spike
Hypothesis: Camarilla pivots (R3/S3) act as strong support/resistance in ranging markets.
Breakout above R3 or below S3 with volume spike indicates institutional participation.
4h EMA50 filter ensures we only trade in direction of higher timeframe trend.
In bull markets: buy breakouts above R3 in uptrend. In bear markets: sell breakdowns below S3 in downtrend.
1h timeframe targets 15-37 trades/year (60-150 over 4 years) by using tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d EMA34 for additional trend confirmation (optional)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivots for 1d (using previous day's OHLC)
    # Camarilla levels: R4 = close + ((high-low)*1.1/2), R3 = close + ((high-low)*1.1/4)
    #               S3 = close - ((high-low)*1.1/4), S4 = close - ((high-low)*1.1/2)
    # We use previous day's data to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values  # previous day high
    prev_low = df_1d['low'].shift(1).values    # previous day low
    prev_close = df_1d['close'].shift(1).values # previous day close
    
    # Calculate Camarilla levels
    camarilla_range = (prev_high - prev_low) * 1.1
    r3 = prev_close + (camarilla_range / 4)
    s3 = prev_close - (camarilla_range / 4)
    
    # Align Camarilla levels to 1h timeframe (they change only once per day)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 50, 34)  # volume MA, 4h EMA50, 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Trend filters
        uptrend_4h = curr_close > ema_50_4h_aligned[i]
        downtrend_4h = curr_close < ema_50_4h_aligned[i]
        uptrend_1d = curr_close > ema_34_1d_aligned[i]
        downtrend_1d = curr_close < ema_34_1d_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above R3 AND uptrend on 4h AND volume spike
            long_entry = (curr_high > r3_aligned[i]) and uptrend_4h and vol_spike
            # Short: price breaks below S3 AND downtrend on 4h AND volume spike
            short_entry = (curr_low < s3_aligned[i]) and downtrend_4h and vol_spike
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price breaks below S3 (reversal) OR loss of 4h uptrend
            if (curr_low < s3_aligned[i]) or (curr_close < ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position management
            # Exit: price breaks above R3 (reversal) OR loss of 4h downtrend
            if (curr_high > r3_aligned[i]) or (curr_close > ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0