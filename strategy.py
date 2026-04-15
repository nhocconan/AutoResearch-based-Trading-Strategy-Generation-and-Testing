#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla pivot breakout with volume confirmation and choppiness regime filter
# Works in bull/bear: Camarilla levels act as support/resistance in all regimes,
# volume confirms institutional participation, chop filter avoids whipsaws in ranging markets
# Target: 30-80 trades over 4 years (7-20/year) to stay within fee-efficient range

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for daily timeframe
    # Camarilla: R4 = Close + ((High-Low) * 1.1/2), R3 = Close + ((High-Low) * 1.1/4)
    #          S3 = Close - ((High-Low) * 1.1/4), S4 = Close - ((High-Low) * 1.1/2)
    # We'll use R3 and S3 as primary breakout levels
    daily_range = daily_high - daily_low
    camarilla_r3 = daily_close + (daily_range * 1.1 / 4)
    camarilla_s3 = daily_close - (daily_range * 1.1 / 4)
    
    # Calculate daily ATR(14) for volatility/chop filter
    tr1 = pd.Series(daily_high - daily_low)
    tr2 = pd.Series(np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr3 = pd.Series(np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]])))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate daily choppiness index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    # Values > 61.8 = ranging, < 38.2 = trending
    atr_sum = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_sum / (max_high - min_low + 1e-10)) / np.log10(14)
    # Handle division by zero and invalid values
    chop_raw = np.where((max_high - min_low) > 0, chop_raw, 50.0)
    chop_raw = np.where(np.isnan(chop_raw), 50.0, chop_raw)
    
    # Align HTF indicators to 1d timeframe (no additional delay needed for these)
    camarilla_r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    atr_14_1d = align_htf_to_ltf(prices, df_1d, atr_14)
    chop_1d = align_htf_to_ltf(prices, df_1d, chop_raw)
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_1d[i]) or np.isnan(camarilla_s3_1d[i]) or 
            np.isnan(atr_14_1d[i]) or np.isnan(chop_1d[i]) or 
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # 1. 1d price breaks above Camarilla R3 with volume confirmation → long
        # 2. 1d price breaks below Camarilla S3 with volume confirmation → short
        # 3. Regime filter: CHOP > 50 (avoid strong trends where breakouts fail)
        # 4. Volume confirmation: volume > 1.5x average
        # 5. Volatility filter: ATR > 0.3% of price (avoid extremely low volatility)
        # 6. Discrete position sizing: 0.25
        
        # Long conditions: 1d breakout above Camarilla R3
        if (close[i] > camarilla_r3_1d[i] and            # 1d price above R3
            chop_1d[i] > 50 and                          # Ranging/weak trend regime (CHOP > 50)
            volume_ratio[i] > 1.5 and                    # Strong volume confirmation
            atr_14_1d[i] > 0.003 * close[i]):            # Minimum volatility filter
            signals[i] = 0.25
            
        # Short conditions: 1d breakdown below Camarilla S3
        elif (close[i] < camarilla_s3_1d[i] and          # 1d price below S3
              chop_1d[i] > 50 and                        # Ranging/weak trend regime (CHOP > 50)
              volume_ratio[i] > 1.5 and                  # Strong volume confirmation
              atr_14_1d[i] > 0.003 * close[i]):          # Minimum volatility filter
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_Volume_Chop_Filter"
timeframe = "1d"
leverage = 1.0