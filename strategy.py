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
    
    # Get weekly HTF data once before loop (for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Calculate weekly EMA(34) for trend filter
    weekly_ema_34 = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align weekly EMA to 6h timeframe
    weekly_ema_34_6h = align_htf_to_ltf(prices, df_1w, weekly_ema_34)
    
    # Get daily HTF data once before loop (for pivot levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily pivot points (Camarilla style)
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4)
    #          S3 = C - ((H-L)*1.1/4), S4 = C - ((H-L)*1.1/2)
    # Where C = (H+L+C)/3 (typical price)
    typical_price = (daily_high + daily_low + daily_close) / 3.0
    daily_range = daily_high - daily_low
    
    r3 = typical_price + (daily_range * 1.1 / 4)
    s3 = typical_price - (daily_range * 1.1 / 4)
    r4 = typical_price + (daily_range * 1.1 / 2)
    s4 = typical_price - (daily_range * 1.1 / 2)
    
    # Align daily Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_ema_34_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(atr_14[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only long when price above weekly EMA34, short when below
        trend_long = close[i] > weekly_ema_34_6h[i]
        trend_short = close[i] < weekly_ema_34_6h[i]
        
        # Entry conditions with discrete sizing (0.25)
        # Long: break above R3 in uptrend OR break above R4 in any trend with volume
        if ((close[i] > r3_6h[i] and trend_long and volume_ratio[i] > 1.5) or
            (close[i] > r4_6h[i] and volume_ratio[i] > 2.0)):
            signals[i] = 0.25
            
        # Short: break below S3 in downtrend OR break below S4 in any trend with volume
        elif ((close[i] < s3_6h[i] and trend_short and volume_ratio[i] > 1.5) or
              (close[i] < s4_6h[i] and volume_ratio[i] > 2.0)):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3_S3_WeeklyEMA_Trend_Filter"
timeframe = "6h"
leverage = 1.0