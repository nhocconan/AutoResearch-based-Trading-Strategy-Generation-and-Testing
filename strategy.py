#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with 1d volume regime filter
# Camarilla R3/S3 levels act as intraday support/resistance. Breakouts with
# above-average volume (1d volume > 20-period MA) indicate institutional participation.
# Works in bull/bear: breakouts capture momentum, volume filter avoids false breakouts in low-liquidity periods.
# Target: 50-150 trades over 4 years (12-37/year) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on prior day)
    # Camarilla uses prior day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels
    # R4 = Close + (High-Low)*1.1/2
    # R3 = Close + (High-Low)*1.1/4
    # S3 = Close - (High-Low)*1.1/4
    # S4 = Close - (High-Low)*1.1/2
    camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 4.0
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4.0
    camarilla_r4 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    camarilla_s4 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate 1d volume ratio (current vs 20-period average) for regime filter
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = df_1d['volume'].values / (vol_ma_20 + 1e-10)
    volume_ratio_6h = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    
    signals = np.zeros(n)
    
    # Session filter: avoid low-volume Asian session (00-08 UTC) for 6h bars
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 23)  # UTC 8:00-23:00 (covers EU/US sessions)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(volume_ratio_6h[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Long conditions:
        # 1. Price breaks above Camarilla R3 (intraday resistance)
        # 2. Volume regime: 1d volume > 1.5x 20-day average (institutional participation)
        # 3. Not already above R4 (avoid chasing extended breakouts)
        if (close[i] > r3_6h[i] and
            volume_ratio_6h[i] > 1.5 and
            close[i] <= r4_6h[i]):
            signals[i] = 0.25
            
        # Short conditions:
        # 1. Price breaks below Camarilla S3 (intraday support)
        # 2. Volume regime: 1d volume > 1.5x 20-day average
        # 3. Not already below S4 (avoid chasing extended breakdowns)
        elif (close[i] < s3_6h[i] and
              volume_ratio_6h[i] > 1.5 and
              close[i] >= s4_6h[i]):
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_1d_CamarillaR3S3_VolumeRegime_v1"
timeframe = "6h"
leverage = 1.0