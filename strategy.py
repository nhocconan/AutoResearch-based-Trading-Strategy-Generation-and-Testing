#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND price > 1d EMA50 (uptrend) AND volume > 1.2x average
# - Short when Williams %R(14) > -20 (overbought) AND price < 1d EMA50 (downtrend) AND volume > 1.2x average
# - Exit when Williams %R returns to -50 (mean reversion midpoint) OR volume drops below average
# - Uses 1d EMA50 for trend alignment to avoid counter-trend trades
# - Volume confirmation filters weak breakouts
# - Targets 12-25 trades/year (50-100 total over 4 years) to avoid fee drag
# - Williams %R is effective at identifying exhaustion points in both bull and bear markets
# - Combined with trend filter, it captures mean reversion within the dominant trend

name = "6h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on 6h data
    highest_high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - prices['close'].values) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.2x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.2 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long: oversold + uptrend + volume spike
            if (williams_r[i] < -80 and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short: overbought + downtrend + volume spike
            elif (williams_r[i] > -20 and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 (mean reversion complete)
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