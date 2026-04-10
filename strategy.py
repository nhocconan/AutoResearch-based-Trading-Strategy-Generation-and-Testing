#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion with 12h trend filter and volume confirmation
# - Williams %R(14) from 6h data: oversold < -80 for long, overbought > -20 for short
# - Trend filter: 12h EMA50 slope (rising/falling) to align with higher timeframe momentum
# - Volume confirmation: current volume > 1.5x 20-period average to avoid false signals
# - Exit: Williams %R crosses above -50 (for longs) or below -50 (for shorts)
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R works well in ranging markets; 12h trend filter prevents counter-trend trades in strong trends

name = "6h_12h_williamsr_meanreversion_volume_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute Williams %R(14) from 6h data
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close_6h) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Pre-compute 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Pre-compute 12h EMA(50) slope for trend direction (rising/falling)
    ema50_12h_slope = np.diff(ema50_12h_aligned, prepend=ema50_12h_aligned[0])
    ema50_12h_rising = ema50_12h_slope > 0
    ema50_12h_falling = ema50_12h_slope < 0
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema50_12h_aligned[i]) or 
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
            # Long when Williams %R oversold (< -80) AND 12h uptrend with volume spike
            if (williams_r[i] < -80 and 
                ema50_12h_rising[i] and 
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when Williams %R overbought (> -20) AND 12h downtrend with volume spike
            elif (williams_r[i] > -20 and 
                  ema50_12h_falling[i] and 
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit when Williams %R crosses -50
            exit_signal = False
            if position == 1:  # Long position
                if williams_r[i] > -50:  # Exit when crosses above -50
                    exit_signal = True
            elif position == -1:  # Short position
                if williams_r[i] < -50:  # Exit when crosses below -50
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