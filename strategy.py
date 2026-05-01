#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R reversal with 1d EMA50 trend filter and volume confirmation.
# Long when %R crosses above -80 from below AND 1d close > EMA50 AND volume > 1.5x 20-bar average.
# Short when %R crosses below -20 from above AND 1d close < EMA50 AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Williams %R identifies overbought/oversold conditions; EMA50 provides trend bias; volume confirmation reduces false signals.
# Works in bull markets (trend continuation from pullbacks) and bear markets (mean reversion at extremes).
# Primary timeframe: 4h, HTF: 1d for trend filter.

name = "4h_WilliamsR_Reversal_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    
    # 1d EMA50 calculation
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 1d close aligned for trend bias
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Williams %R calculation on 4h data (period=14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_aligned[i]) or np.isnan(close_1d_aligned[i]) or \
           np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
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
        wr_cross_up = williams_r[i] > -80 and williams_r[i-1] <= -80  # cross above -80
        wr_cross_down = williams_r[i] < -20 and williams_r[i-1] >= -20  # cross below -20
        
        # Trend filter: use 1d close vs its EMA50 for bias
        bullish_bias = close_1d_aligned[i] > ema_aligned[i]  # 1d close above its EMA50 = bullish
        bearish_bias = close_1d_aligned[i] < ema_aligned[i]  # 1d close below its EMA50 = bearish
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: %R crosses above -80 FROM BELOW AND bullish bias AND volume confirmation
            if (wr_cross_up and 
                bullish_bias and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: %R crosses below -20 FROM ABOVE AND bearish bias AND volume confirmation
            elif (wr_cross_down and 
                  bearish_bias and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: %R crosses above -20 (overbought) OR bearish bias (trend change)
            if (williams_r[i] > -20 and williams_r[i-1] <= -20) or \
               bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: %R crosses below -80 (oversold) OR bullish bias (trend change)
            if (williams_r[i] < -80 and williams_r[i-1] >= -80) or \
               bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals