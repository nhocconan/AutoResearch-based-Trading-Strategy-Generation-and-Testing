#!/usr/bin/env python3
"""
1h RSI Pullback with 4h Trend and 1d Volume Filter
Hypothesis: RSI(14) pullbacks to 40-60 range during strong 4h trends (EMA20 > EMA50) 
provide high-probability continuation entries. 1d volume > 1.5x average confirms 
institutional participation. Designed for 1h to balance trade frequency and signal 
quality, using 4h for trend direction and 1d for regime filter to avoid chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_pullback_4h_trend_1d_volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False).mean().values
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False).mean().values
    trend_up_4h = align_htf_to_ltf(prices, df_4h, ema20_4h > ema50_4h)
    trend_down_4h = align_htf_to_ltf(prices, df_4h, ema20_4h < ema50_4h)
    
    # 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = df_1d['volume'] > (vol_ma_1d * 1.5)
    vol_filter_1h = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    # 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(trend_up_4h[i]) or np.isnan(trend_down_4h[i]) or
            np.isnan(vol_filter_1h[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) or trend reversal
            if rsi[i] > 70 or not trend_up_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) or trend reversal
            if rsi[i] < 30 or not trend_down_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: RSI pullback to 40-60 in uptrend with volume confirmation
            if (40 <= rsi[i] <= 60 and 
                trend_up_4h[i] and 
                vol_filter_1h[i]):
                position = 1
                signals[i] = 0.20
            # Short: RSI pullback to 40-60 in downtrend with volume confirmation
            elif (40 <= rsi[i] <= 60 and 
                  trend_down_4h[i] and 
                  vol_filter_1h[i]):
                position = -1
                signals[i] = -0.20
    
    return signals