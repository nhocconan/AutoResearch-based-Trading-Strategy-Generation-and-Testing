#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot (R3/S3) breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels identify intraday support/resistance. Breakouts above R3 or below S3 with
# volume spike and 1d EMA34 trend alignment capture strong moves. Designed for 12-30 trades/year
# on 12h to minimize fee drag while maintaining edge in bull/bear markets via trend filtering.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Need at least 3 bars for Camarilla calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous day's data (need high, low, close from prior day)
        # For 12h timeframe, we approximate using rolling window of 2 bars (1 day = 2x12h bars)
        if i >= 2:
            prev_high = np.max(high[i-2:i])  # High of previous 12h bar (prior day half)
            prev_low = np.min(low[i-2:i])    # Low of previous 12h bar
            prev_close = close[i-1]          # Close of previous 12h bar
            
            # Camarilla levels
            range_val = prev_high - prev_low
            if range_val > 0:
                camarilla_r3 = prev_close + (range_val * 1.1 / 4)
                camarilla_s3 = prev_close - (range_val * 1.1 / 4)
                
                # Volume confirmation: current volume > 2x EMA20 volume
                vol_ema_20 = pd.Series(volume[max(0, i-19):i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1] if i >= 19 else volume[i]
                volume_spike = volume[i] > (2.0 * vol_ema_20)
                
                breakout_long = close[i] > camarilla_r3 and volume_spike
                breakout_short = close[i] < camarilla_s3 and volume_spike
                
                if position == 0:
                    # Long: break above R3 in 1d uptrend with volume spike
                    if breakout_long and ema_34_1d_aligned[i] > close[i]:
                        signals[i] = 0.25
                        position = 1
                    # Short: break below S3 in 1d downtrend with volume spike
                    elif breakout_short and ema_34_1d_aligned[i] < close[i]:
                        signals[i] = -0.25
                        position = -1
                elif position == 1:
                    # Exit long: price crosses below R3 or loses 1d uptrend
                    if close[i] < camarilla_r3 or ema_34_1d_aligned[i] < close[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                elif position == -1:
                    # Exit short: price crosses above S3 or loses 1d downtrend
                    if close[i] > camarilla_s3 or ema_34_1d_aligned[i] > close[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
            else:
                # Invalid range, flatten if in position
                if position != 0:
                    signals[i] = 0.0
                    position = 0
        else:
            # Not enough data for Camarilla calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals