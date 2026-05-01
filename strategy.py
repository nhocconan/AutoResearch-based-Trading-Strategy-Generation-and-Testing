#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA34 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 (oversold) AND 1d EMA34 rising AND volume > 1.8x 20-bar average.
# Short when Williams %R crosses below -20 (overbought) AND 1d EMA34 falling AND volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25. Session filter 08-20 UTC.
# Target: 50-150 total trades over 4 years (12-37/year) for BTC/ETH/SOL.
# Williams %R provides timely reversal signals in ranging markets.
# 1d EMA34 offers smooth trend alignment with reduced whipsaw.
# Volume spike ensures only high-conviction reversals are traded.

name = "12h_WilliamsR_EMA34_VolumeSpike_v1"
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
    
    # Pre-compute session hours for efficiency (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA34 calculation
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # 1d EMA34 slope (rising/falling)
    ema_34_slope = np.diff(ema_34_aligned, prepend=ema_34_aligned[0])
    ema_34_rising = ema_34_slope > 0
    ema_34_falling = ema_34_slope < 0
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 12h volume > 1.8x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # warmup for Williams %R and EMA
    
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
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.8)
        
        # Williams %R reversal signals
        wr_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80  # cross above -80 (oversold)
        wr_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20  # cross below -20 (overbought)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: WR crosses above -80 AND 1d EMA34 rising AND volume confirmation
            if (wr_cross_up and 
                ema_34_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: WR crosses below -20 AND 1d EMA34 falling AND volume confirmation
            elif (wr_cross_down and 
                  ema_34_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: WR crosses below -50 (momentum loss) OR 1d EMA34 falls (trend change)
            if (williams_r[i] < -50 and williams_r[i-1] >= -50) or \
               ema_34_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: WR crosses above -50 (momentum loss) OR 1d EMA34 rises (trend change)
            if (williams_r[i] > -50 and williams_r[i-1] <= -50) or \
               ema_34_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals