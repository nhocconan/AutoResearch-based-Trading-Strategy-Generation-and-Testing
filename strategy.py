#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: Daily Williams %R mean reversion with weekly EMA50 trend filter
    # Williams %R identifies overbought/oversold conditions for mean reversion
    # Weekly EMA50 filters for long-term trend direction (works in bull/bear)
    # Entry: Williams %R crosses above -80 (oversold) in uptrend or below -20 (overbought) in downtrend
    # Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below + price above weekly EMA50 (uptrend)
            if i > 0 and williams_r[i-1] <= -80 and williams_r[i] > -80 and close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above + price below weekly EMA50 (downtrend)
            elif i > 0 and williams_r[i-1] >= -20 and williams_r[i] < -20 and close[i] < ema50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
            if position == 1:
                if i > 0 and williams_r[i-1] < -50 and williams_r[i] >= -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if i > 0 and williams_r[i-1] > -50 and williams_r[i] <= -50:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Williams_%R_MeanReversion_1wEMA50_Trend_v1"
timeframe = "1d"
leverage = 1.0