#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme reversal with 1w EMA50 trend filter and volume spike confirmation.
# Long when %R < -80 (oversold) with 1w EMA50 uptrend and volume > 1.8x 20-bar average.
# Short when %R > -20 (overbought) with 1w EMA50 downtrend and volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 1d timeframe to capture multi-day reversals.
# Works in bull (buy oversold dips) and bear (sell overbought rallies) via trend filter.

name = "1d_WilliamsR_Extreme_1wEMA50_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for EMA50 and Williams %R calculation
    
    for i in range(start_idx, n):
        # Session filter: 00-23 UTC (trade all sessions for 1d timeframe)
        hour = hours[i]
        
        if np.isnan(ema_50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_ema_50_1w = ema_50_1w_aligned[i]
        
        # Calculate Williams %R from previous 14 periods (need 14 bars of data)
        if i < 14 + start_idx:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[i-14:i])  # highest high over last 14 periods
        lowest_low = np.min(low[i-14:i])     # lowest low over last 14 periods
        
        if highest_high == lowest_low:
            signals[i] = 0.0
            continue
            
        williams_r = -100 * (highest_high - curr_close) / (highest_high - lowest_low)
        
        # Volume confirmation: current 1d volume > 1.8x 20-period average
        if i < 20 + start_idx:
            signals[i] = 0.0
            continue
            
        vol_ma = np.mean(volume[i-20:i])  # 20-period simple moving average
        if vol_ma <= 0:
            signals[i] = 0.0
            continue
        volume_confirm = curr_vol > (vol_ma * 1.8)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND price > 1w EMA50 AND volume confirmation
            if (williams_r < -80 and 
                curr_close > curr_ema_50_1w and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) AND price < 1w EMA50 AND volume confirmation
            elif (williams_r > -20 and 
                  curr_close < curr_ema_50_1w and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R > -20 (overbought) OR price < 1w EMA50 (trend violation)
            if (williams_r > -20 or 
                curr_close < curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R < -80 (oversold) OR price > 1w EMA50 (trend violation)
            if (williams_r < -80 or 
                curr_close > curr_ema_50_1w):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals