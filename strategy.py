#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Bollinger Band squeeze breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when price breaks above upper Bollinger Band(20,2) AND close > 1w EMA50 AND volume > 2.0 * avg_volume(20) on 12h
# Short when price breaks below lower Bollinger Band(20,2) AND close < 1w EMA50 AND volume > 2.0 * avg_volume(20) on 12h
# Exit when price returns to middle Bollinger Band (20-period SMA) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Bollinger Band squeeze identifies low volatility periods primed for breakout
# 1w EMA50 filters for primary trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakdowns in downtrend)

name = "12h_BollingerSqueeze_Breakout_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Bollinger Bands(20,2) on 12h
    close_s = pd.Series(close)
    sma20 = close_s.rolling(window=20, min_periods=20).mean().values
    std20 = close_s.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + (2 * std20)
    lower_bb = sma20 - (2 * std20)
    middle_bb = sma20  # 20-period SMA for exit
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(sma20[i]) or np.isnan(std20[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Bollinger Band, above 1w EMA50, volume confirmation, in session
            if (close[i] > upper_bb[i] and close[i-1] <= upper_bb[i-1] and 
                close[i] > ema50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Bollinger Band, below 1w EMA50, volume confirmation, in session
            elif (close[i] < lower_bb[i] and close[i-1] >= lower_bb[i-1] and 
                  close[i] < ema50_1w_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to middle Bollinger Band OR volume drops below average
            if close[i] <= middle_bb[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to middle Bollinger Band OR volume drops below average
            if close[i] >= middle_bb[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals