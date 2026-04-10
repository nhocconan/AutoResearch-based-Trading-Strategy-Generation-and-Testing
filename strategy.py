#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R < -80 (oversold) AND 1w EMA50 rising AND volume > 1.8x 20-bar avg
# - Short when Williams %R > -20 (overbought) AND 1w EMA50 falling AND volume > 1.8x 20-bar avg
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses 1w EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)
# - Williams %R works well in ranging markets; trend filter adds directional bias in trends

name = "12h_1w_williamsr_meanreversion_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R from daily data (14-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_1d) / (highest_high - lowest_low) * -100
    # Replace division by zero with -50 (neutral)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Align Williams %R to LTF
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
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
            # Long when Williams %R < -80 (oversold) AND 1w uptrend with volume spike
            if (williams_r_aligned[i] < -80 and 
                prices['close'].iloc[i] > ema50_1w_aligned[i] and  # price above 1w EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when Williams %R > -20 (overbought) AND 1w downtrend with volume spike
            elif (williams_r_aligned[i] > -20 and 
                  prices['close'].iloc[i] < ema50_1w_aligned[i] and  # price below 1w EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to Williams %R = -50 (mean reversion)
            # Exit when Williams %R returns to -50 level
            exit_signal = False
            if position == 1:  # Long position
                if williams_r_aligned[i] >= -50:
                    exit_signal = True
            elif position == -1:  # Short position
                if williams_r_aligned[i] <= -50:
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