#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1w EMA200 trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels on 6h timeframe.
# Long: Williams %R crosses above -80 from below + price above 1w EMA200 + volume > 1.5x avg volume
# Short: Williams %R crosses below -20 from above + price below 1w EMA200 + volume > 1.5x avg volume
# Trend filter uses 1w EMA200 to ensure alignment with major trend.
# Volume confirmation reduces false signals.
# Works in both bull and bear markets by using 1w EMA200 as trend filter.
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 60-minute data for Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We'll use 14-period lookback
    lookback = 14
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback, n):
        highest_high[i] = np.max(high[i-lookback:i])
        lowest_low[i] = np.min(low[i-lookback:i])
    
    williams_r = np.full(n, np.nan)
    for i in range(lookback, n):
        if highest_high[i] != lowest_low[i]:  # Avoid division by zero
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100
        else:
            williams_r[i] = -50  # Neutral when range is zero
    
    # 1-week EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Average volume (14-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(lookback, n):
        avg_volume[i] = np.mean(volume[i-lookback:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(lookback, n):
        # Skip if any required data is not ready
        if (np.isnan(williams_r[i]) or np.isnan(williams_r[i-1]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema_200_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below + above EMA200 + volume confirmation
            if (wr > -80 and wr_prev <= -80 and  # Cross above -80
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Williams %R crosses below -20 from above + below EMA200 + volume confirmation
            elif (wr < -20 and wr_prev >= -20 and  # Cross below -20
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) or below EMA200
            if (wr > -20 or  # Overbought exit
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) or above EMA200
            if (wr < -80 or  # Oversold exit
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_WilliamsR_EMA200_Volume"
timeframe = "6h"
leverage = 1.0