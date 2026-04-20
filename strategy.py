# 12h_Camarilla_R3S3_Reversal_Volume_Confirmation_v1
# Hypothesis: In ranging markets, price tends to reverse from extreme Camarilla levels (R3/S3). 
# In trending markets, price continues from R1/S1 levels. Volume confirms institutional participation.
# Uses 1d Camarilla levels, 12h EMA20 for trend filter, and volume spike for confirmation.
# Designed to work in both bull (trend continuation) and bear (mean reversion from extremes) markets.
# Target: 20-50 trades over 4 years to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_Reversal_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === 12h: EMA20 for trend direction ===
    close_12h = prices['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 1d: Calculate Camarilla pivot levels ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    r2_1d = close_1d + (high_1d - low_1d) * 1.1 / 6
    s2_1d = close_1d - (high_1d - low_1d) * 1.1 / 6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    r4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2
    s4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align all Camarilla levels
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 12h: ATR(14) for volatility ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_20_12h[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or
            np.isnan(r2) or np.isnan(s2) or np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or
            np.isnan(s4) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.5x 24-period 12h average volume ===
        if i >= 24:
            vol_ma = np.mean(prices['volume'].iloc[i-24:i].values)
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long conditions:
            # 1. Price closes above R3 with volume (break of strong resistance -> potential uptrend)
            # 2. OR price bounces from S3 with volume (support hold in ranging market)
            if ((current_close > r3 and vol_condition) or 
                (current_close < s3 and current_close > s4 and vol_condition and current_close > ema_trend)):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price closes below S3 with volume (break of strong support -> potential downtrend)
            # 2. OR price reverses from R3 with volume (resistance hold in ranging market)
            elif ((current_close < s3 and vol_condition) or 
                  (current_close > r3 and current_close < r4 and vol_condition and current_close < ema_trend)):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below R1 (failure to hold support)
            # 2. Price reaches R4 (extreme overbought - take profit)
            # 3. ATR-based stop loss
            if (current_close < r1 or
                current_close >= r4 or
                current_close < entry_price - 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above S1 (failure to hold resistance)
            # 2. Price reaches S4 (extreme oversold - take profit)
            # 3. ATR-based stop loss
            if (current_close > s1 or
                current_close <= s4 or
                current_close > entry_price + 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals