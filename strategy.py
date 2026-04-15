#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Calculate weekly Donchian channels (20-period) for trend
    highest_20w = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_20w = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ATR(14) for volatility filter
    tr1 = pd.Series(weekly_high - weekly_low)
    tr2 = pd.Series(np.abs(weekly_high - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr3 = pd.Series(np.abs(weekly_low - np.concatenate([[weekly_close[0]], weekly_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14w = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to daily timeframe with proper delay
    highest_20w_aligned = align_htf_to_ltf(prices, df_1w, highest_20w)
    lowest_20w_aligned = align_htf_to_ltf(prices, df_1w, lowest_20w)
    atr_14w_aligned = align_htf_to_ltf(prices, df_1w, atr_14w)
    
    # Calculate daily Donchian breakout (20-period) for entry signals
    highest_20d = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20d = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20d + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20w_aligned[i]) or np.isnan(lowest_20w_aligned[i]) or 
            np.isnan(atr_14w_aligned[i]) or np.isnan(highest_20d[i]) or 
            np.isnan(lowest_20d[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. Weekly trend filter: price above weekly Donchian high = bullish bias
        #    price below weekly Donchian low = bearish bias
        # 2. Daily Donchian breakout with volume confirmation
        # 3. Volatility filter: avoid extremely low volatility
        # 4. Discrete position sizing: 0.25
        
        # Long conditions: bullish weekly trend + daily breakout above upper band
        if (close[i] > highest_20w_aligned[i] and      # Weekly uptrend
            close[i] > highest_20d[i] and              # Daily breakout above 20-day high
            volume_ratio[i] > 1.5 and                  # Strong volume confirmation
            atr_14w_aligned[i] > 0.003 * close[i]):    # Minimum volatility filter
            signals[i] = 0.25
            
        # Short conditions: bearish weekly trend + daily breakdown below lower band
        elif (close[i] < lowest_20w_aligned[i] and   # Weekly downtrend
              close[i] < lowest_20d[i] and           # Daily breakdown below 20-day low
              volume_ratio[i] > 1.5 and              # Strong volume confirmation
              atr_14w_aligned[i] > 0.003 * close[i]): # Minimum volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_WeeklyDonchianTrend_DailyDonchianBreakout_Volume"
timeframe = "1d"
leverage = 1.0