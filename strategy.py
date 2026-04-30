#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R with 1w EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions on daily timeframe
# 1w EMA34 provides primary trend direction to avoid counter-trend trades
# Volume spike (1.5x 20-period average) confirms institutional participation
# Only trade when %R crosses below -80 (oversold) in uptrend or above -20 (overbought) in downtrend
# Discrete sizing 0.25 minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).
# Works in both bull and bear markets by following higher timeframe trend.

name = "1d_WilliamsR_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop (MTF Rule #1)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Williams %R needs 14-period lookback
    start_idx = 14
    
    for i in range(start_idx, n):
        # Calculate Williams %R (14-period)
        highest_high = np.max(high[i-13:i+1])  # highest high in last 14 periods including current
        lowest_low = np.min(low[i-13:i+1])    # lowest low in last 14 periods including current
        
        if highest_high == lowest_low:
            williams_r = -50.0  # avoid division by zero
        else:
            williams_r = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        
        # Volume confirmation: volume > 1.5x 20-period average
        vol_ma_20 = np.mean(volume[max(0, i-20):i])
        volume_spike = volume[i] > (1.5 * vol_ma_20)
        
        curr_ema_34_1w = ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike
            if volume_spike:
                # Bullish entry: Williams %R crosses below -80 (oversold) AND price above 1w EMA34 (uptrend)
                if williams_r < -80 and close[i] > curr_ema_34_1w:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Williams %R crosses above -20 (overbought) AND price below 1w EMA34 (downtrend)
                elif williams_r > -20 and close[i] < curr_ema_34_1w:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -20 (overbought) or price falls below 1w EMA34
            if williams_r > -20 or close[i] < curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -80 (oversold) or price rises above 1w EMA34
            if williams_r < -80 or close[i] > curr_ema_34_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals