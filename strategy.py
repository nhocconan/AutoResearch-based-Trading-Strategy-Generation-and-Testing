#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme reversal with 1d EMA34 trend filter and volume spike confirmation.
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short).
# Trend filter: price > 1d EMA34 for longs, price < 1d EMA34 for shorts.
# Volume confirmation: current 4h volume > 2.0x 20-bar 4h volume average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed to work in both bull (buy dips) and bear (sell rallies).

name = "4h_WilliamsR_Extreme_1dEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA34
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Williams %R(14) calculation on 4h data (requires 14 periods of high/low/close)
        if i < 14 + start_idx:  # need extra warmup for Williams %R
            signals[i] = 0.0
            continue
            
        # Calculate Williams %R: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
        highest_high = np.max(high[i-13:i+1])  # 14 periods including current
        lowest_low = np.min(low[i-13:i+1])
        if highest_high == lowest_low:  # avoid division by zero
            williams_r = -50.0  # neutral
        else:
            williams_r = ((highest_high - curr_close) / (highest_high - lowest_low)) * -100
        
        # Volume confirmation: current 4h volume > 2.0x 20-period average from 4h HTF data
        df_4h = get_htf_data(prices, '4h')
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND price > 1d EMA34 AND volume confirmation
            if (williams_r < -80.0 and 
                curr_close > curr_ema_34_1d and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1d EMA34 AND volume confirmation
            elif (williams_r > -20.0 and 
                  curr_close < curr_ema_34_1d and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 1d EMA34 (trend violation)
            if (williams_r > -20.0 or 
                curr_close < curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR price > 1d EMA34 (trend violation)
            if (williams_r < -80.0 or 
                curr_close > curr_ema_34_1d):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals