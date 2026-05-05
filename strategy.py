#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R(14) < -80 = oversold (long setup), > -20 = overbought (short setup)
# Enter long when %R crosses above -80 from below AND price > 1d EMA34 AND volume > 1.5x 20-period average
# Enter short when %R crosses below -20 from above AND price < 1d EMA34 AND volume > 1.5x 20-period average
# Exit when %R crosses opposite extreme (-20 for long, -80 for short) OR trend filter fails
# Uses discrete sizing (0.25) to limit fee drag. Target: 15-35 trades/year per symbol.
# Williams %R captures momentum exhaustion, EMA34 filters for primary trend to avoid counter-trend whipsaws,
# Volume confirmation ensures institutional participation. Works in bull markets via longs in uptrends
# and bear markets via shorts in downtrends, with mean-reversion entries during extreme readings.

name = "6h_WilliamsR_EXTREME_1dEMA34_VolumeSpike_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 1d data for Williams %R calculation (need high, low, close)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Volume confirmation: volume > 1.5x 20-period average (spike filter)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Williams %R crosses above -80 (from below) AND price > EMA34 AND volume spike
            williams_r_now = williams_r_aligned[i]
            williams_r_prev = williams_r_aligned[i-1]
            price_above_ema = close[i] > ema_34_aligned[i]
            
            if (williams_r_now > -80 and williams_r_prev <= -80 and 
                price_above_ema and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Williams %R crosses below -20 (from above) AND price < EMA34 AND volume spike
            elif (williams_r_now < -20 and williams_r_prev >= -20 and 
                  close[i] < ema_34_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 (overbought) OR price < EMA34 (trend fail)
            williams_r_now = williams_r_aligned[i]
            williams_r_prev = williams_r_aligned[i-1]
            price_below_ema = close[i] < ema_34_aligned[i]
            
            if (williams_r_now < -20 and williams_r_prev >= -20) or price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 (oversold) OR price > EMA34 (trend fail)
            williams_r_now = williams_r_aligned[i]
            williams_r_prev = williams_r_aligned[i-1]
            price_above_ema = close[i] > ema_34_aligned[i]
            
            if (williams_r_now > -80 and williams_r_prev <= -80) or price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals