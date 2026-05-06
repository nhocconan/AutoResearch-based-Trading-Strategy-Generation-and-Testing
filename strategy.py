#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Camarilla pivot levels (R1, S1) with 1d EMA34 trend filter and volume spike confirmation
# Long when price touches or breaks above 1d Camarilla S1 level AND 1d EMA34 > EMA89 AND volume > 2.0 * avg_volume(20)
# Short when price touches or breaks below 1d Camarilla R1 level AND 1d EMA34 < EMA89 AND volume > 2.0 * avg_volume(20)
# Exit when price reaches 1d Camarilla midpoint (M) or opposite level (R1 for long, S1 for short)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# 1d Camarilla provides strong intraday support/resistance levels proven to work on ETH/USDT
# 1d EMA filter ensures alignment with daily trend, reducing counter-trend trades
# Volume spike confirmation filters weak breakouts and captures institutional interest
# Works in bull (buying dips at S1 in uptrend) and bear (selling rallies at R1 in downtrend)

name = "4h_1dCamarilla_R1S1_1dEMATrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Camarilla levels and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # M = (H+L+C)/3 (pivot point)
    # We'll use R1, S1, and M (midpoint)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    # Set first values to avoid roll issues
    prev_close_1d[0] = close_1d[0]
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_s1 = prev_close_1d - (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_r1 = prev_close_1d + (prev_high_1d - prev_low_1d) * 1.1 / 12
    camarilla_m = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d bar)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_m_aligned = align_htf_to_ltf(prices, df_1d, camarilla_m)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_m_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price touches or breaks above Camarilla S1 with 1d EMA34 > EMA89 and volume spike
            if (close[i] >= camarilla_s1_aligned[i] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price touches or breaks below Camarilla R1 with 1d EMA34 < EMA89 and volume spike
            elif (close[i] <= camarilla_r1_aligned[i] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches Camarilla midpoint M or drops below S1 (profit take or reversal)
            if close[i] >= camarilla_m_aligned[i] or close[i] <= camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches Camarilla midpoint M or rises above R1 (profit take or reversal)
            if close[i] <= camarilla_m_aligned[i] or close[i] >= camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals