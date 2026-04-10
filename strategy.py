#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1d close > 1d EMA200 (uptrend) AND volume > 1.5x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) AND 1d close < 1d EMA200 (downtrend) AND volume > 1.5x 20-bar avg
# - Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# - Uses 1d EMA200 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 30-50 trades/year on 4h timeframe (120-200 total over 4 years)
# - Williams %R is effective for mean reversion in ranging markets; trend filter adds directional bias

name = "4h_1d_williamsr_meanreversion_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Pre-compute Williams %R(14) on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(volume_20_avg[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new mean reversion entries
            # Long when Williams %R < -80 (oversold) AND 1d uptrend with volume spike
            if (williams_r_aligned[i] < -80 and 
                prices['close'].iloc[i] > ema200_1d_aligned[i] and  # price above 1d EMA200
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when Williams %R > -20 (overbought) AND 1d downtrend with volume spike
            elif (williams_r_aligned[i] > -20 and 
                  prices['close'].iloc[i] < ema200_1d_aligned[i] and  # price below 1d EMA200
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Williams %R crosses -50
            exit_signal = False
            if position == 1:  # Long position
                if williams_r_aligned[i] > -50:  # Exit when no longer oversold
                    exit_signal = True
            elif position == -1:  # Short position
                if williams_r_aligned[i] < -50:  # Exit when no longer overbought
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals