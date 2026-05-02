#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume spike
# Uses 4h/1d for signal direction (trend/volume regime) and 1h only for entry timing precision
# Targets 15-37 trades/year (60-150 total over 4 years) to minimize fee drag
# Camarilla levels provide institutional pivot points with proven effectiveness
# 4h EMA34 determines trend bias; volume spike confirms participation
# Works in bull via breakouts with trend, bear via fade of false breakouts at key levels
# Discrete position sizing: 0.20 (20% of capital) balances exposure and risk

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 1h Camarilla levels (prior completed 1h bar's range)
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4, etc.
    # Simplified: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    # Using prior completed bar: shift(1)
    hl_range = pd.Series(high - low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    camarilla_r3 = prev_close + 1.1 * hl_range * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * hl_range * 1.1 / 4
    
    # Calculate 4h EMA34 trend (prior completed 4h bar's EMA)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    ema_34 = pd.Series(df_4h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_4h, ema_34)
    
    # Calculate 1d volume regime (prior completed 1d bar's volume average)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().shift(1).values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    volume_spike = volume > (vol_ma_20_aligned * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 4h EMA34 (bullish bias) AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: price breaks below Camarilla S3 AND price < 4h EMA34 (bearish bias) AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Camarilla S3 OR below 4h EMA34 (trend change)
            if close[i] < camarilla_s3[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: price rises above Camarilla R3 OR above 4h EMA34 (trend change)
            if close[i] > camarilla_r3[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals