#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R + 1w trend filter + volume confirmation
# - Long when 1d Williams %R < -80 (oversold) AND 1w close > 1w EMA50 (uptrend) AND volume > 1.5x 20-day average
# - Short when 1d Williams %R > -20 (overbought) AND 1w close < 1w EMA50 (downtrend) AND volume > 1.5x 20-day average
# - Exit when Williams %R crosses -50 (mean reversion) with volume confirmation
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe
# - Williams %R identifies exhaustion points in ranging markets
# - 1w EMA50 filter ensures we trade with the weekly trend
# - Volume confirmation ensures signals have conviction

name = "1d_1w_williamsr_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d Williams %R (14)
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
    
    # Pre-compute 1d volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: Williams %R oversold AND weekly uptrend AND volume spike
            if (williams_r[i] < -80 and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: Williams %R overbought AND weekly downtrend AND volume spike
            elif (williams_r[i] > -20 and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when Williams %R crosses -50 (mean reversion) with volume confirmation
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