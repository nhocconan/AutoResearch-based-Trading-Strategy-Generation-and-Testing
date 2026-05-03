#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation
# Camarilla pivot levels identify intraday support/resistance that often act as breakout points.
# Breakouts above R3 or below S3 with volume confirmation and aligned weekly trend capture
# strong moves while minimizing false signals. Weekly EMA50 filter ensures we only trade
# in the direction of the higher timeframe trend. Designed for 10-25 trades/year on 1d
# to minimize fee drag while maintaining edge in both bull and bear markets.

name = "1d_Camarilla_R3S3_1wEMA50_VolumeSpike"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start after sufficient warmup for Camarilla calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels using previous day's OHLC
        if i >= 1:
            # Use previous bar's OHLC (daily timeframe)
            prev_close = close[i-1]
            prev_high = high[i-1]
            prev_low = low[i-1]
            
            # Camarilla levels
            range_val = prev_high - prev_low
            if range_val <= 0:
                # Skip if invalid range
                if position != 0:
                    signals[i] = 0.0
                    position = 0
                continue
                
            # Resistance levels
            r3 = prev_close + range_val * 1.1 / 4
            r4 = prev_close + range_val * 1.1 / 2
            
            # Support levels
            s3 = prev_close - range_val * 1.1 / 4
            s4 = prev_close - range_val * 1.1 / 2
            
            # Volume confirmation: current volume > 2x 20-period volume EMA
            vol_start = max(0, i-19)
            vol_ema_20 = pd.Series(volume[vol_start:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
            volume_spike = volume[i] > (2.0 * vol_ema_20)
            
            # Breakout conditions
            breakout_long = close[i] > r3 and volume_spike
            breakout_short = close[i] < s3 and volume_spike
            
            if position == 0:
                # Long: break above R3 in 1w uptrend with volume spike
                if breakout_long and ema_50_1w_aligned[i] > close[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: break below S3 in 1w downtrend with volume spike
                elif breakout_short and ema_50_1w_aligned[i] < close[i]:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long: price crosses below R3 or loses 1w uptrend
                if close[i] < r3 or ema_50_1w_aligned[i] < close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above S3 or loses 1w downtrend
                if close[i] > s3 or ema_50_1w_aligned[i] > close[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Not enough data for Camarilla calculation yet
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals