#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator system with 1w trend filter and volume confirmation
# - Uses Williams Alligator (JAW=13, TEETH=8, LIPS=5) on 1d to identify trend direction
# - Long when LIPS > TEETH > JAW (bullish alignment) AND price > LIPS AND 1w close > 1w EMA50 AND volume > 1.5x 20-bar average
# - Short when LIPS < TEETH < JAW (bearish alignment) AND price < LIPS AND 1w close < 1w EMA50 AND volume > 1.5x 20-bar average
# - Exit when Alligator lines cross (LIPS crosses TEETH) or volume drops below average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag
# - Williams Alligator is effective in both trending and ranging markets by showing convergence/divergence
# - 1w EMA50 filter ensures alignment with higher timeframe trend
# - Volume confirmation reduces false signals

name = "1d_williams_alligator_volume_trend_v1"
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
    
    # Pre-compute 1d Williams Alligator
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    
    # Jaw (13-period SMMA of median price)
    median_price = (high_1d + low_1d) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values
    
    # Teeth (8-period SMMA of typical price)
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    teeth = pd.Series(typical_price).rolling(window=8, min_periods=8).mean().values
    
    # Lips (5-period SMMA of weighted close)
    weighted_close = (high_1d + low_1d + 2 * close_1d) / 4.0
    lips = pd.Series(weighted_close).rolling(window=5, min_periods=5).mean().values
    
    # Pre-compute 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Pre-compute volume confirmation: > 1.5x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.5 * volume_20_avg)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(13, n):  # Start after Alligator warmup
        # Skip if any required data is invalid
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_20_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw
            bullish_align = lips[i] > teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw
            bearish_align = lips[i] < teeth[i] < jaw[i]
            
            # Long signal: bullish alignment AND price > Lips AND 1w uptrend AND volume spike
            if (bullish_align and 
                prices['close'].iloc[i] > lips[i] and 
                prices['close'].iloc[i] > ema_50_1w_aligned[i] and
                vol_spike.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: bearish alignment AND price < Lips AND 1w downtrend AND volume spike
            elif (bearish_align and 
                  prices['close'].iloc[i] < lips[i] and 
                  prices['close'].iloc[i] < ema_50_1w_aligned[i] and
                  vol_spike.iloc[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit conditions:
            # 1. Alligator lines cross (Lips crosses Teeth)
            # 2. Volume drops below average (loss of momentum)
            if position == 1:  # Long position
                if (lips[i] <= teeth[i]) or (not vol_spike.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25  # Hold long
            elif position == -1:  # Short position
                if (lips[i] >= teeth[i]) or (not vol_spike.iloc[i]):
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25  # Hold short
    
    return signals