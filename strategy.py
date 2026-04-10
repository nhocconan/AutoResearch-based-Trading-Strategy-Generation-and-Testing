#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + volume confirmation + 1w trend filter (EMA200)
# - Long when Williams %R < -80 (oversold) AND volume > 1.5x 20-period average AND price > 1w EMA200 (bullish bias)
# - Short when Williams %R > -20 (overbought) AND volume > 1.5x 20-period average AND price < 1w EMA200 (bearish bias)
# - Exit when Williams %R crosses -50 (mean reversion midpoint) with volume confirmation
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R identifies exhaustion points in ranging markets
# - Volume confirmation ensures breakouts have conviction
# - 1w EMA200 filter ensures we trade with the weekly trend, reducing counter-trend whipsaws

name = "6h_1w_williamsr_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Pre-compute 6h Williams %R (14)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 6h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA200 trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after warmup for EMA200
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_200_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND volume spike AND price > 1w EMA200
            if (williams_r[i] < -80 and 
                volume_spike[i] and 
                close[i] > ema_200_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND volume spike AND price < 1w EMA200
            elif (williams_r[i] > -20 and 
                  volume_spike[i] and 
                  close[i] < ema_200_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R crosses -50 (mean reversion midpoint) with volume confirmation
            exit_long = (position == 1 and 
                        williams_r[i] > -50 and 
                        volume_spike[i])
            exit_short = (position == -1 and 
                         williams_r[i] < -50 and 
                         volume_spike[i])
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals