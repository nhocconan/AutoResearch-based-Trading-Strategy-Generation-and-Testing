# 1d_Long_Term_Trend_With_Weekly_Filter
# Hypothesis: Long-term trend following on daily timeframe with weekly trend filter.
# Uses 50-day EMA for trend direction and 200-day EMA for long-term bias.
# Weekly trend filter ensures we only trade in the direction of the weekly trend.
# Entry when price crosses above/below 50-day EMA with volume confirmation.
# Exit when price crosses back below/above 50-day EMA.
# Designed to capture major trends while avoiding counter-trend trades.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 50-day EMA for trend direction
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 200-day EMA for long-term bias
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Weekly trend filter: 20-week EMA on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: current > 1.5x 20-day median volume
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):  # Start after warmup for 50-day EMA
        # Skip if any required data is NaN
        if (np.isnan(ema_50[i]) or np.isnan(ema_200[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price above 50-day EMA, above 200-day EMA (long-term bias), 
        # weekly trend up (price above weekly EMA), volume spike
        if (close[i] > ema_50[i] and 
            close[i] > ema_200[i] and 
            close[i] > ema_20_1w_aligned[i] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price below 50-day EMA, below 200-day EMA (long-term bias),
        # weekly trend down (price below weekly EMA), volume spike
        elif (close[i] < ema_50[i] and 
              close[i] < ema_200[i] and 
              close[i] < ema_20_1w_aligned[i] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses back below/above 50-day EMA
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < ema_50[i]) or
               (signals[i-1] == -0.25 and close[i] > ema_50[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "1d_Long_Term_Trend_With_Weekly_Filter"
timeframe = "1d"
leverage = 1.0