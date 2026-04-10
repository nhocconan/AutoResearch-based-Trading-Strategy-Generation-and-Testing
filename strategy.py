#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R mean reversion with 1d trend filter and volume confirmation
# - Long when Williams %R < -80 (oversold) AND 1d EMA50 rising AND volume > 1.8x 20-bar avg
# - Short when Williams %R > -20 (overbought) AND 1d EMA50 falling AND volume > 1.8x 20-bar avg
# - Exit when Williams %R crosses -50 (mean reversion to midpoint)
# - Uses 1d EMA50 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 20-35 trades/year on 4h timeframe (80-140 total over 4 years)
# - Williams %R is effective in ranging markets; trend filter adds directional bias

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
    
    # Pre-compute Williams %R (14-period) from 4h data
    high_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    low_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    close = prices['close'].values
    williams_r = -100 * (high_14 - close) / (high_14 - low_14)
    # Handle division by zero when high == low
    williams_r = np.where((high_14 - low_14) == 0, -50, williams_r)
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Pre-compute volume confirmation: > 1.8x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.8 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or 
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
            # Long when oversold AND 1d uptrend with volume spike
            if (williams_r[i] < -80 and 
                prices['close'].iloc[i] > ema50_1d_aligned[i] and  # price above 1d EMA50
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when overbought AND 1d downtrend with volume spike
            elif (williams_r[i] > -20 and 
                  prices['close'].iloc[i] < ema50_1d_aligned[i] and  # price below 1d EMA50
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to midpoint (-50)
            # Exit when Williams %R crosses -50 (mean reversion)
            exit_signal = False
            if position == 1:  # Long position
                if williams_r[i] >= -50:
                    exit_signal = True
            elif position == -1:  # Short position
                if williams_r[i] <= -50:
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