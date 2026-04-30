#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume spike confirmation.
# Long when Williams %R(14) < -80 (oversold) AND price > 12h EMA50 (uptrend) AND volume > 1.8x 24-bar average.
# Short when Williams %R(14) > -20 (overbought) AND price < 12h EMA50 (downtrend) AND volume > 1.8x 24-bar average.
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 100-200 total trades over 4 years (25-50/year). Williams %R provides mean reversion edge in ranging markets,
# while 12h EMA50 filter ensures we only trade with the higher timeframe trend, reducing false signals in strong trends.

name = "6h_WilliamsR_MeanRev_12hEMA50_Trend_VolumeSpike_v1"
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
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R(14) on 6h timeframe
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: volume > 1.8x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.8 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 24)  # warmup for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_williams_r = williams_r[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: oversold (Williams %R < -80), uptrend (price > 12h EMA50), volume confirmation
            if (curr_williams_r < -80 and 
                curr_close > ema_50_12h_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought (Williams %R > -20), downtrend (price < 12h EMA50), volume confirmation
            elif (curr_williams_r > -20 and 
                  curr_close < ema_50_12h_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Williams %R crosses back above -50 (exiting oversold territory)
            if curr_williams_r > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Williams %R crosses back below -50 (exiting overbought territory)
            if curr_williams_r < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals