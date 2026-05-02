#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with daily trend filter and volume confirmation
# Camarilla pivot levels identify key intraday support/resistance; breakout above R3 or below S3 with
# volume spike (2.0x 20-period average) and daily EMA(34) trend filter captures strong momentum moves
# while avoiding false breakouts. Designed for 12h timeframe to target 12-37 trades/year (50-150 total).
# Works in bull markets via breakout continuation and in bear markets via filtered short breakdowns.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (high + low + close) / 3
    # Camarilla levels: R4 = close + (high - low) * 1.1/2, R3 = close + (high - low) * 1.1/4, etc.
    # We use previous day's data to avoid look-ahead
    typical_price = (high + low + close) / 3.0
    # Shift by 1 to use previous day's typical price for today's levels
    typical_price_prev = pd.Series(typical_price).shift(1).values
    high_prev = pd.Series(high).shift(1).values
    low_prev = pd.Series(low).shift(1).values
    
    # Calculate Camarilla R3 and S3 levels from previous day
    rangeprev = high_prev - low_prev
    R3 = typical_price_prev + rangeprev * 1.1 / 4.0
    S3 = typical_price_prev - rangeprev * 1.1 / 4.0
    
    # Calculate volume spike (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Camarilla calculation and volume MA)
    start_idx = 20  # buffer for 20-period volume MA and 1-day shift
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Camarilla R3 + 1d close > daily EMA + volume spike
            if (close[i] > R3[i] and 
                close[i] > ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 + 1d close < daily EMA + volume spike
            elif (close[i] < S3[i] and 
                  close[i] < ema_1d_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price drops below Camarilla S3 (reversal signal) or daily trend breaks
            if close[i] < S3[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 (reversal signal) or daily trend breaks
            if close[i] > R3[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals