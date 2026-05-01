#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R extreme reversal with 1w EMA50 trend filter and volume spike confirmation
# Williams %R measures overbought/oversold: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
# Extreme readings: %R < -80 (oversold) or %R > -20 (overbought)
# Long: %R crosses above -80 from below + price above 1w EMA50 + volume spike
# Short: %R crosses below -20 from above + price below 1w EMA50 + volume spike
# Uses 1d timeframe for lower frequency (target: 7-25 trades/year) to minimize fee drag
# Works in bull markets via buying oversold dips in uptrend and in bear markets via selling overbought rallies in downtrend

name = "1d_WilliamsR_Extreme_Reversal_1wEMA50_Trend_VolumeSpike_v1"
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
    
    # 1w HTF data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams %R calculation (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(50, 14, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1w EMA50
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Williams %R extremes and crossovers
        wr_current = williams_r[i]
        wr_previous = williams_r[i-1]
        
        # Long conditions: %R crosses above -80 from below (bullish reversal)
        long_signal = (wr_previous < -80 and wr_current >= -80) and vol_spike and uptrend
        
        # Short conditions: %R crosses below -20 from above (bearish reversal)
        short_signal = (wr_previous > -20 and wr_current <= -20) and vol_spike and downtrend
        
        if position == 0:  # Flat - look for new entries
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on %R crossing above -20 (overbought) or trend reversal
            if wr_current >= -20 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on %R crossing below -80 (oversold) or trend reversal
            if wr_current <= -80 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals