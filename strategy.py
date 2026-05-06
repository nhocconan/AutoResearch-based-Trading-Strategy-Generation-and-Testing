#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Camarilla pivot breakout with 1w EMA34 trend filter and volume confirmation
# Long when price breaks above 1w Camarilla R3 level AND 1w EMA34 > EMA89 AND volume > 2.0 * avg_volume(20)
# Short when price breaks below 1w Camarilla S3 level AND 1w EMA34 < EMA89 AND volume > 2.0 * avg_volume(20)
# Exit when price touches 1w Camarilla pivot point (PP) or opposite S1/R1 level
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe
# 1w Camarilla provides strong institutional support/resistance levels
# 1w EMA34/EMA89 filter ensures alignment with weekly trend, reducing counter-trend trades
# High volume confirmation (2.0x) filters weak breakouts
# Works in bull (trend continuation breakouts above R3) and bear (trend continuation breakdowns below S3)

name = "1d_1wCamarilla_R3S3_Breakout_1wEMA34Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for Camarilla pivots and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 89:  # Need sufficient data for EMA89
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (based on previous week's OHLC)
    # Camarilla formulas: PP = (H+L+C)/3, R4 = C + ((H-L)*1.1/2), S4 = C - ((H-L)*1.1/2)
    # We use R3/S3 and PP, R1, S1 for entries/exits
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Camarilla levels
    pp_1w = typical_price_1w
    r1_1w = close_1w + (range_1w * 1.1 / 12)
    s1_1w = close_1w - (range_1w * 1.1 / 12)
    r2_1w = close_1w + (range_1w * 1.1 / 6)
    s2_1w = close_1w - (range_1w * 1.1 / 6)
    r3_1w = close_1w + (range_1w * 1.1 / 4)
    s3_1w = close_1w - (range_1w * 1.1 / 4)
    r4_1w = close_1w + (range_1w * 1.1 / 2)
    s4_1w = close_1w - (range_1w * 1.1 / 2)
    
    # Align 1w Camarilla levels to 1d timeframe (wait for completed 1w bar)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    
    # Calculate 1w EMA34 and EMA89 for trend filter
    close_series_1w = pd.Series(close_1w)
    ema_34_1w = close_series_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1w = close_series_1w.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Align 1w EMA values to 1d timeframe (wait for completed 1w bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_aligned = align_htf_to_ltf(prices, df_1w, ema_89_1w)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 1w Camarilla R3 with 1w EMA34 > EMA89 and volume confirmation
            if (close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1] and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1w Camarilla S3 with 1w EMA34 < EMA89 and volume confirmation
            elif (close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1] and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches 1w Camarilla pivot point (PP) or S1 level (profit take or reversal)
            if close[i] <= pp_aligned[i] or close[i] <= s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches 1w Camarilla pivot point (PP) or R1 level (profit take or reversal)
            if close[i] >= pp_aligned[i] or close[i] >= r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals