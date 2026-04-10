#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND weekly close > weekly EMA20 (uptrend) AND volume > 1.3x average
# - Short when Williams %R(14) > -20 (overbought) AND weekly close < weekly EMA20 (downtrend) AND volume > 1.3x average
# - Exit when Williams %R returns to -50 (mean reversion) OR volume drops below average
# - Williams %R captures extreme momentum reversals that work in both bull and bear markets
# - Weekly trend filter ensures we trade with the higher timeframe momentum
# - Volume confirmation reduces false signals
# - Targets 20-35 trades/year (80-140 total over 4 years) to balance opportunity and fee drag

name = "1d_1w_williamsr_meanreversion_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) on daily data
    highest_high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    lowest_low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - prices['close'].values) / (highest_high_14 - lowest_low_14)
    
    # Pre-compute 1w EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute volume filter: < average volume for exit
    vol_normal = prices['volume'] < volume_20_avg
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for mean reversion entries
            # Long: oversold (-80) with volume spike AND weekly uptrend
            if (williams_r[i] < -80 and 
                vol_spike.iloc[i] and 
                prices['close'].iloc[i] > ema20_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short: overbought (-20) with volume spike AND weekly downtrend
            elif (williams_r[i] > -20 and 
                  vol_spike.iloc[i] and 
                  prices['close'].iloc[i] < ema20_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Williams %R returns to -50 (mean reversion complete)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (williams_r[i] > -50 or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (williams_r[i] < -50 or 
                    vol_normal.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals