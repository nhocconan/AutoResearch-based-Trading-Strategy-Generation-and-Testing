#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
# Williams %R(14) identifies overbought/oversold conditions; reversals from extreme levels
# provide high-probability mean reversion entries. 1d EMA34 ensures trades align with
# intermediate-term trend to avoid counter-trend whipsaw. Volume spike (>2x 20-bar average)
# confirms conviction. Discrete sizing 0.25 manages drawdown. Session filter 08-20 UTC
# avoids low-liquidity hours. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_WilliamsR_1dEMA34_VolumeSpike_Reversal_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d trend: price above/below EMA34
    price_above_ema = close > ema_34_aligned
    price_below_ema = close < ema_34_aligned
    
    # Williams %R(14) on 6h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 6h volume > 2x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R, EMA and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade only 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        if np.isnan(williams_r[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Williams %R reversal signals
        # Long: Williams %R crosses above -80 from oversold
        # Short: Williams %R crosses below -20 from overbought
        wr_long = williams_r[i] > -80 and williams_r[i-1] <= -80
        wr_short = williams_r[i] < -20 and williams_r[i-1] >= -20
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R reversal from oversold AND price > 1d EMA34 AND volume confirmation
            if (wr_long and 
                price_above_ema[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R reversal from overbought AND price < 1d EMA34 AND volume confirmation
            elif (wr_short and 
                  price_below_ema[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR price < 1d EMA34 (trend change)
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or \
               not price_above_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR price > 1d EMA34 (trend change)
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or \
               not price_below_ema[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals