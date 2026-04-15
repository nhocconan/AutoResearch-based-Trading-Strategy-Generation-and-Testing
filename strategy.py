#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band Squeeze + Volume + 1d Trend Filter
# Long when price breaks above upper BB with volume spike and 1d EMA50 uptrend
# Short when price breaks below lower BB with volume spike and 1d EMA50 downtrend
# Bollinger squeeze (low volatility) precedes breakouts, effective in both bull and bear
# Uses discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Bollinger Bands (20, 2)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    bb_period = 20
    bb_mid = pd.Series(close_12h).rolling(window=bb_period, min_periods=bb_period).mean()
    bb_std = pd.Series(close_12h).rolling(window=bb_period, min_periods=bb_period).std()
    bb_upper = (bb_mid + 2 * bb_std).values
    bb_lower = (bb_mid - 2 * bb_std).values
    bb_upper_aligned = align_htf_to_ltf(prices, df_12h, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_12h, bb_lower)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Bollinger Band width (volatility measure) for squeeze detection
    bb_width = bb_upper - bb_lower
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width)
    
    # Squeeze condition: BB width below 20-period mean (low volatility)
    bb_width_ma = pd.Series(bb_width_aligned).rolling(window=20, min_periods=1).mean()
    squeeze = bb_width_aligned < bb_width_ma.values
    
    # Volume confirmation: current > 2.0x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_upper_aligned[i]) or np.isnan(bb_lower_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(squeeze[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: price breaks above upper BB + squeeze + volume + 1d uptrend
        if (close[i] > bb_upper_aligned[i] and squeeze[i] and 
            volume[i] > vol_threshold[i] and close[i] > ema_1d_aligned[i]):
            signals[i] = 0.25
        
        # Short: price breaks below lower BB + squeeze + volume + 1d downtrend
        elif (close[i] < bb_lower_aligned[i] and squeeze[i] and 
              volume[i] > vol_threshold[i] and close[i] < ema_1d_aligned[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle band or volatility expands
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] <= bb_mid.iloc[i] if hasattr(bb_mid, 'iloc') else bb_mid[i])) or
               (signals[i-1] == -0.25 and (close[i] >= bb_mid.iloc[i] if hasattr(bb_mid, 'iloc') else bb_mid[i])))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Bollinger_Squeeze_Volume_Trend"
timeframe = "12h"
leverage = 1.0