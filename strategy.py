#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R with 1d EMA34 trend filter and volume confirmation
# - Long when Williams %R crosses above -20 (oversold bounce) with price above 1d EMA34 and volume expansion
# - Short when Williams %R crosses below -80 (overbought rejection) with price below 1d EMA34 and volume expansion
# - Exit when Williams %R crosses back below -50 for longs or above -50 for shorts
# - Volume filter requires current volume > 1.2x 20-period average
# - Williams %R calculated on 1d timeframe to reduce noise and avoid whipsaws in 6h chart
# - Designed to capture mean reversion in ranging markets while respecting higher timeframe trend
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WilliamsR_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R and EMA calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filters (6h timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.2 * vol_ma_20)  # Volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(williams_r_6h[i]) or np.isnan(ema_34_1d_6h[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r_6h[i]
        wr_prev = williams_r_6h[i-1] if i > 0 else -50
        
        if position == 0:
            # Long entry: Williams %R crosses above -20 (oversold bounce) with volume and above EMA34
            if wr > -20 and wr_prev <= -20 and volume_filter[i] and close[i] > ema_34_1d_6h[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Williams %R crosses below -80 (overbought rejection) with volume and below EMA34
            elif wr < -80 and wr_prev >= -80 and volume_filter[i] and close[i] < ema_34_1d_6h[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50
            if wr < -50 and wr_prev >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50
            if wr > -50 and wr_prev <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals