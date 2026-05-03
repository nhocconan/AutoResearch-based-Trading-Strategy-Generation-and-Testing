#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation.
# Long: Close breaks above Camarilla R3 AND 12h close > 12h EMA50 (uptrend) AND volume > 2.0x 20-period MA
# Short: Close breaks below Camarilla S3 AND 12h close < 12h EMA50 (downtrend) AND volume > 2.0x 20-period MA
# Exit: Opposite Camarilla break (R4/S4) or trend reversal (close crosses EMA50) or volume drops below average.
# Uses discrete sizing 0.25. Target: 50-150 total trades over 4 years (12-37/year).
# Camarilla levels provide precise intraday support/resistance; 12h EMA50 filters for higher-timeframe trend;
# volume confirmation reduces false breakouts. Works in bull via longs and bear via shorts when aligned with 12h trend.

name = "6h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (EMA50) and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 12h Camarilla levels (using typical price)
    typical_price_12h = (df_12h['high'].values + df_12h['low'].values + df_12h['close'].values) / 3.0
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Camarilla levels: based on previous day's range
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    range_12h = high_12h - low_12h
    camarilla_r3_12h = typical_price_12h + 1.1 * range_12h
    camarilla_s3_12h = typical_price_12h - 1.1 * range_12h
    camarilla_r4_12h = typical_price_12h + 1.5 * range_12h
    camarilla_s4_12h = typical_price_12h - 1.5 * range_12h
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    camarilla_r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4_12h)
    camarilla_s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4_12h)
    
    # Volume regime: current 6h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_r3_12h_aligned[i]) or 
            np.isnan(camarilla_s3_12h_aligned[i]) or np.isnan(camarilla_r4_12h_aligned[i]) or 
            np.isnan(camarilla_s4_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 12h trend
        is_uptrend = close_val > ema_trend  # Using 6h close vs 12h EMA50 for timely signal
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: Close breaks above Camarilla R3 AND uptrend AND volume spike
            if close_val > camarilla_r3_12h_aligned[i] and is_uptrend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below Camarilla S3 AND downtrend AND volume spike
            elif close_val < camarilla_s3_12h_aligned[i] and is_downtrend and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close breaks below Camarilla S4 OR trend reverses (close < EMA50) OR volume drops
            if (close_val < camarilla_s4_12h_aligned[i] or 
                close_val < ema_trend or 
                not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close breaks above Camarilla R4 OR trend reverses (close > EMA50) OR volume drops
            if (close_val > camarilla_r4_12h_aligned[i] or 
                close_val > ema_trend or 
                not vol_spike):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals