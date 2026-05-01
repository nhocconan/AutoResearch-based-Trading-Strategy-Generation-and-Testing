#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA trend filter and volume confirmation.
# Long when Williams %R(14) crosses above -80 (oversold reversal) AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-bar average.
# Short when Williams %R(14) crosses below -20 (overbought reversal) AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 50-150 total trades over 4 years (12-37/year).
# Williams %R captures mean reversals in ranging markets while EMA50 filters for trend alignment.
# Primary timeframe: 6h, HTF: 1d for EMA trend and Williams %R calculation.

name = "6h_WilliamsR_EMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA and Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams %R(14) on 1d data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min()
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Calculate EMA50 on 1d close
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        if np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_aligned[i]) or \
           np.isnan(vol_ma[i]):
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
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume spike threshold
        
        # Williams %R reversal signals
        williams_r_prev = williams_r_aligned[i-1] if i > 0 else -50
        williams_r_curr = williams_r_aligned[i]
        
        # Bullish reversal: Williams %R crosses above -80 from below
        bullish_reversal = (williams_r_prev <= -80) and (williams_r_curr > -80)
        # Bearish reversal: Williams %R crosses below -20 from above
        bearish_reversal = (williams_r_prev >= -20) and (williams_r_curr < -20)
        
        # Trend filter: price relative to 1d EMA50
        above_ema = curr_close > ema_50_aligned[i]
        below_ema = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R bullish reversal AND above EMA50 AND volume confirmation
            if (bullish_reversal and 
                above_ema and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R bearish reversal AND below EMA50 AND volume confirmation
            elif (bearish_reversal and 
                  below_ema and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses below -50 (momentum loss) OR price crosses below EMA50
            if (williams_r_curr < -50 or 
                curr_close < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses above -50 (momentum loss) OR price crosses above EMA50
            if (williams_r_curr > -50 or 
                curr_close > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals