#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long: Close breaks above Camarilla R3 AND price > 1d EMA34 AND volume > 1.5x 20-period MA
# Short: Close breaks below Camarilla S3 AND price < 1d EMA34 AND volume > 1.5x 20-period MA
# Exit: Opposite Camarilla breakout or price crosses 1d EMA34 or volume drops.
# Discrete sizing 0.25. Target: 75-200 total trades over 4 years (19-50/year).
# Camarilla provides intraday support/resistance; 1d EMA34 filters for trend alignment; volume confirmation reduces false breakouts.
# Works in bull via long signals and bear via short signals when aligned with higher timeframe trend.

name = "4h_Camarilla_R3S3_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels based on previous day's OHLC
    # We need to get the previous day's OHLC for each 4h bar
    # Since we're on 4h timeframe, we'll use the prior 1d bar's OHLC
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(n):
        if i == 0:
            continue
        # Get the 1d index corresponding to the current 4h bar's time
        # We'll use the prior completed 1d bar for Camarilla calculation
        # This is simplified - in practice we'd need to map 4h time to 1d bar
        # For now, we'll use a rolling window approach on 1d data aligned to 4h
        pass
    
    # Simpler approach: Calculate Camarilla from 1d OHLC and align to 4h
    # This gives us the Camarilla levels for each 1d bar, then we align to 4h
    if len(df_1d) >= 1:
        # Calculate Camarilla for each 1d bar
        camarilla_r3_1d = np.full(len(df_1d), np.nan)
        camarilla_s3_1d = np.full(len(df_1d), np.nan)
        
        for j in range(len(df_1d)):
            if j == 0:
                continue  # Need prior day for calculation
            # Prior day's OHLC
            ph = df_1d['high'].iloc[j-1]
            pl = df_1d['low'].iloc[j-1]
            pc = df_1d['close'].iloc[j-1]
            
            # Camarilla formulas
            camarilla_r3_1d[j] = pc + (ph - pl) * 1.1 / 4
            camarilla_s3_1d[j] = pc - (ph - pl) * 1.1 / 4
        
        # Align Camarilla levels to 4h timeframe
        camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
        camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    
    # Volume regime: current 4h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if close_val > camarilla_r3[i] and close_val > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif close_val < camarilla_s3[i] and close_val < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Camarilla S3 OR price < 1d EMA34 OR volume drops
            if close_val < camarilla_s3[i] or close_val < ema_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Camarilla R3 OR price > 1d EMA34 OR volume drops
            if close_val > camarilla_r3[i] or close_val > ema_val or not vol_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals