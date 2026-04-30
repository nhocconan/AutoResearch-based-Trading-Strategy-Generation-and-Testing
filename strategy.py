#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) with 1d uptrend (close > 1d EMA50) and volume > 1.8x 24-bar avg.
# Short when Williams %R > -20 (overbought) with 1d downtrend (close < 1d EMA50) and volume > 1.8x 24-bar avg.
# Exit on opposite Williams %R level touch (mean reversion within the pivot structure).
# Williams %R is effective in ranging markets and captures reversals in bear rallies.
# 1d EMA50 provides longer-term trend filter, reducing false signals in choppy markets.
# Timeframe: 12h, HTF: 1d as per experiment guidelines.

name = "12h_WilliamsR_MeanRev_1dEMA50_Trend_VolumeConfirmation_v1"
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
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R calculation (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.8x 24-period average (balanced to avoid overtrading)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Williams %R and EMA50
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_ema_50_1d = ema_50_1d_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold), uptrend (close > 1d EMA50), volume spike
            if (curr_williams_r < -80 and 
                curr_close > curr_ema_50_1d and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), downtrend (close < 1d EMA50), volume spike
            elif (curr_williams_r > -20 and 
                  curr_close < curr_ema_50_1d and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R touches -20 (overbought, mean reversion)
            if curr_williams_r >= -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R touches -80 (oversold, mean reversion)
            if curr_williams_r <= -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals