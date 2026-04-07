#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# Williams %R measures momentum overbought/oversold levels
# Long when Williams %R crosses above -80 from below in 1d uptrend with volume confirmation
# Short when Williams %R crosses below -20 from above in 1d downtrend with volume confirmation
# Exit when Williams %R crosses opposite threshold or stoploss at 2.5 * ATR
# Williams %R formula: (Highest High - Close) / (Highest High - Lowest Low) * -100
# Uses daily trend to filter for stronger trends that work in both bull and bear markets
# Target: 80-180 total trades over 4 years (20-45/year)

name = "6h_williamsr_1d_trend_vol_v1"
timeframe = "6h"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) and EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Average volume for volume confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_avg[i]) or np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses above -20 (overbought)
            elif williams_r[i] > -20 and williams_r[i-1] <= -20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R crosses below -80 (oversold)
            elif williams_r[i] < -80 and williams_r[i-1] >= -80:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Trend filter: 1d EMA(50) > EMA(200) for uptrend, < for downtrend
            uptrend = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
            downtrend = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
            
            # Volume confirmation: current volume > 1.8 * average volume
            volume_confirm = volume[i] > 1.8 * vol_avg[i]
            
            # Long: Williams %R crosses above -80 from below in uptrend with volume
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Williams %R crosses below -20 from above in downtrend with volume
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals