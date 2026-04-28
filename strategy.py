#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA50 trend filter and volume spike confirmation
# Williams %R(14) < -80 = oversold, > -20 = overbought
# Long when %R crosses above -80 from below AND price > 1d EMA50 (uptrend) AND volume > 2x 20-bar avg
# Short when %R crosses below -20 from above AND price < 1d EMA50 (downtrend) AND volume > 2x 20-bar avg
# Uses extreme readings to catch reversals in both bull and bear markets, filtered by 1d trend
# Target: 12-35 trades/year via strict extreme thresholds and trend/volume filters
# Works in bull markets (buying oversold in uptrend) and bear markets (selling overbought in downtrend)

name = "6h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where(highest_high == lowest_low, -50, williams_r)
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(40, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        price = close[i]
        ema_50 = ema_50_1d_aligned[i]
        wr = williams_r[i]
        wr_prev = williams_r[i-1] if i > 0 else -50
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Williams %R crosses above -80 from below (ending oversold)
            # AND price > 1d EMA50 (uptrend) AND volume confirmation
            if wr > -80 and wr_prev <= -80 and price > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Williams %R crosses below -20 from above (ending overbought)
            # AND price < 1d EMA50 (downtrend) AND volume confirmation
            elif wr < -20 and wr_prev >= -20 and price < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Williams %R crosses below -50 (momentum loss) or trend breaks
            if wr < -50 or price < ema_50 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Williams %R crosses above -50 (momentum loss) or trend breaks
            if wr > -50 or price > ema_50 or not vol_conf:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals