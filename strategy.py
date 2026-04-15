#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Camarilla R1/S1 breakout with weekly trend filter and volume confirmation
# Works in bull/bear: Weekly trend ensures we trade with higher timeframe momentum,
# Camarilla levels provide intraday support/resistance, volume filters breakout validity.
# Target: 30-100 trades over 4 years on 1d timeframe (discreet entries to avoid fee drag).

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
    if len(df_1w < 50):
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Calculate weekly EMA34 for trend filter
    weekly_close_series = pd.Series(weekly_close)
    ema34_1w = weekly_close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_volume = df_1d['volume'].values
    
    # Calculate Camarilla levels (based on previous day)
    # R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12
    # R2 = C + (H-L)*1.1/6
    # S2 = C - (H-L)*1.1/6
    # R3 = C + (H-L)*1.1/4
    # S3 = C - (H-L)*1.1/4
    # R4 = C + (H-L)*1.1/2
    # S4 = C - (H-L)*1.1/2
    diff = daily_high - daily_low
    r1 = daily_close + diff * 1.1 / 12
    s1 = daily_close - diff * 1.1 / 12
    r2 = daily_close + diff * 1.1 / 6
    s2 = daily_close - diff * 1.1 / 6
    r3 = daily_close + diff * 1.1 / 4
    s3 = daily_close - diff * 1.1 / 4
    r4 = daily_close + diff * 1.1 / 2
    s4 = daily_close - diff * 1.1 / 2
    
    # Align daily Camarilla levels to 1d timeframe (no additional delay needed)
    r1_1d = align_htf_to_ltf(prices, df_1d, r1)
    s1_1d = align_htf_to_ltf(prices, df_1d, s1)
    r2_1d = align_htf_to_ltf(prices, df_1d, r2)
    s2_1d = align_htf_to_ltf(prices, df_1d, s2)
    r3_1d = align_htf_to_ltf(prices, df_1d, r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, s3)
    r4_1d = align_htf_to_ltf(prices, df_1d, r4)
    s4_1d = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = daily_volume / (vol_ma_20 + 1e-10)
    volume_ratio_1d = align_htf_to_ltf(prices, df_1d, volume_ratio)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(atr_14_1d[i]) or np.isnan(volume_ratio_1d[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: Price breaks above R1 with weekly uptrend, volume, and volatility
        # Short: Price breaks below S1 with weekly downtrend, volume, and volatility
        # Discrete position sizing: 0.25
        
        # Weekly trend: price above EMA34 = uptrend, below = downtrend
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        # Long conditions
        if (close[i] > r1_1d[i] and              # Break above R1
            weekly_uptrend and                   # Weekly uptrend filter
            volume_ratio_1d[i] > 1.5 and         # Volume confirmation
            atr_14_1d[i] > 0.01 * close[i]):     # Volatility filter (ATR > 1% of price)
            signals[i] = 0.25
            
        # Short conditions
        elif (close[i] < s1_1d[i] and            # Break below S1
              weekly_downtrend and               # Weekly downtrend filter
              volume_ratio_1d[i] > 1.5 and       # Volume confirmation
              atr_14_1d[i] > 0.01 * close[i]):   # Volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_WeeklyEMA34_Volume_Filter"
timeframe = "1d"
leverage = 1.0