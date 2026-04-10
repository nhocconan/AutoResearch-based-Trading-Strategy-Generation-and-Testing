#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R mean reversion with 1w trend filter and volume confirmation
# - Long when Williams %R(14) < -80 (oversold) AND 1w EMA200 rising AND volume > 1.5x 20-bar avg
# - Short when Williams %R(14) > -20 (overbought) AND 1w EMA200 falling AND volume > 1.5x 20-bar avg
# - Exit when Williams %R returns to -50 (mean reversion to equilibrium)
# - Uses 1w EMA200 for trend filter to avoid counter-trend trades
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-25 trades/year on 12h timeframe (50-100 total over 4 years)
# - Williams %R works well in ranging markets; trend filter adds directional bias in trends

name = "12h_1w_williamsr_meanreversion_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(200) for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Pre-compute Williams %R(14) from 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema200_1w_aligned[i]) or np.isnan(williams_r[i]) or 
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
            if (williams_r[i] < -80 and 
                close[i] > ema200_1w_aligned[i] and  # price above 1w EMA200
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short when Williams %R > -20 (overbought) AND 1w downtrend with volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema200_1w_aligned[i] and  # price below 1w EMA200
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit to mean reversion (-50)
            # Exit when Williams %R returns to -50
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