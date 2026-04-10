#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND daily close > daily EMA50 AND volume > 1.2x average
# - Short when Williams %R(14) > -20 (overbought) AND daily close < daily EMA50 AND volume > 1.2x average
# - Exit when Williams %R returns to -50 level or volume drops below average
# - Williams %R captures short-term extremes in ranging markets; trend filter ensures alignment with major trend
# - Volume confirmation prevents false signals
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Williams %R is effective in both bull and bear markets for mean reversion during pullbacks

name = "12h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute Williams %R(14) on 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          -100 * (highest_high - close) / (highest_high - lowest_low), 
                          -50)  # Default to neutral when range is zero
    
    # Pre-compute volume confirmation: > 1.2x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.2 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(williams_r[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long entry: Williams %R oversold (< -80) with volume spike AND daily uptrend
            if (williams_r[i] < -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Williams %R overbought (> -20) with volume spike AND daily downtrend
            elif (williams_r[i] > -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to neutral (-50) level
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (williams_r[i] >= -50 or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (williams_r[i] <= -50 or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals