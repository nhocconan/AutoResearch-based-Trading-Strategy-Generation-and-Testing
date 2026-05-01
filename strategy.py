#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1w EMA trend filter and volume confirmation.
# Uses weekly EMA50 to identify primary trend direction, reducing counter-trend trades.
# Long when Williams %R crosses above -80 (oversold reversal) AND price > weekly EMA50 AND volume > 1.5x 20-bar average.
# Short when Williams %R crosses below -20 (overbought reversal) AND price < weekly EMA50 AND volume > 1.5x 20-bar average.
# Williams %R is calculated on 6h data with 14-period lookback.
# Designed to capture mean reversals within the prevailing weekly trend, working in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "6h_WilliamsR_Reversal_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA calculation
        return np.zeros(n)
    
    # Weekly EMA50 calculation
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R calculation on 6h data (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(williams_r[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Williams %R reversal signals
        wr_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80  # cross above -80 (oversold)
        wr_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20  # cross below -20 (overbought)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND price > weekly EMA50 AND volume confirmation
            if (wr_cross_up and 
                curr_close > ema_50_1w_aligned[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND price < weekly EMA50 AND volume confirmation
            elif (wr_cross_down and 
                  curr_close < ema_50_1w_aligned[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR price crosses below weekly EMA50
            if (williams_r[i] > -20 and williams_r[i-1] <= -20) or \
               (curr_close < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR price crosses above weekly EMA50
            if (williams_r[i] < -80 and williams_r[i-1] >= -80) or \
               (curr_close > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals