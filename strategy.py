#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1-day trend filter and volume confirmation
# Williams %R measures overbought/oversold levels (-80 to -20). 
# Long when: Williams %R < -80 (oversold) AND 1-day EMA50 > EMA200 (bullish trend) AND volume > 1.5x 20-period average
# Short when: Williams %R > -20 (overbought) AND 1-day EMA50 < EMA200 (bearish trend) AND volume > 1.5x 20-period average
# Exit when Williams %R crosses back above -50 (for longs) or below -50 (for shorts)
# Target: 20-40 trades/year per symbol. Williams %R is effective in ranging markets which occur frequently in crypto.
name = "4h_WilliamsR_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 and EMA200 on daily data
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Determine trend: 1 for bullish (EMA50 > EMA200), -1 for bearish (EMA50 < EMA200)
    trend_1d = np.where(ema50_1d > ema200_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Williams %R calculation (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, period, 20)  # Wait for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        wr = williams_r[i]
        trend = trend_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: oversold + bullish trend + volume confirmation
            if wr < -80 and trend == 1 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: overbought + bearish trend + volume confirmation
            elif wr > -20 and trend == -1 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Williams %R crosses back above -50 (momentum fading)
            if wr > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Williams %R crosses back below -50 (momentum fading)
            if wr < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals