#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d VWAP trend filter and volume confirmation
# Camarilla pivot levels identify key support/resistance from prior day's range
# 1d VWAP provides institutional trend bias to avoid counter-trend trades
# Volume confirmation ensures breakouts have institutional participation
# Target: 75-200 total trades over 4 years (19-50/year) with disciplined entries
name = "4h_Camarilla_1dVWAP_Volume"
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
    
    # 1d VWAP for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily VWAP: cumulative(price * volume) / cumulative(volume)
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap_1d = (vwap_numerator / vwap_denominator).values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Camarilla pivot levels from previous day
    # Calculate for each 4h bar using previous day's OHLC
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_r2 = np.full(n, np.nan)
    camarilla_r1 = np.full(n, np.nan)
    camarilla_s1 = np.full(n, np.nan)
    camarilla_s2 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    # We need previous day's data for each point
    # Since we have daily data, we'll use the prior completed day's values
    prev_day_high = np.roll(df_1d['high'].values, 1)
    prev_day_low = np.roll(df_1d['low'].values, 1)
    prev_day_close = np.roll(df_1d['close'].values, 1)
    
    # Handle first day
    prev_day_high[0] = prev_day_low[0] = prev_day_close[0] = df_1d['close'].iloc[0]
    
    # Calculate Camarilla levels
    range_ = prev_day_high - prev_day_low
    camarilla_r4 = prev_day_close + range_ * 1.1 / 2
    camarilla_r3 = prev_day_close + range_ * 1.1 / 4
    camarilla_r2 = prev_day_close + range_ * 1.1 / 6
    camarilla_r1 = prev_day_close + range_ * 1.1 / 12
    camarilla_s1 = prev_day_close - range_ * 1.1 / 12
    camarilla_s2 = prev_day_close - range_ * 1.1 / 6
    camarilla_s3 = prev_day_close - range_ * 1.1 / 4
    camarilla_s4 = prev_day_close - range_ * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and above 1d VWAP
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_confirm[i] and 
                close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume confirmation and below 1d VWAP
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_confirm[i] and 
                  close[i] < vwap_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or goes below VWAP
            if (close[i] < camarilla_s1_aligned[i]) or (close[i] < vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or goes above VWAP
            if (close[i] > camarilla_r1_aligned[i]) or (close[i] > vwap_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals