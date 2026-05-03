#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume confirmation.
# Uses 1w EMA50 to filter trend direction, 1d Camarilla levels for breakout entries.
# Long when price breaks above R3 with volume > 1.5x 20-period MA and close > 1w EMA50 (uptrend).
# Short when price breaks below S3 with volume spike and close < 1w EMA50 (downtrend).
# Discrete sizing 0.25. Target: 30-100 total trades over 4 years (7-25/year).
# Camarilla levels provide institutional support/resistance; 1w EMA50 filters counter-trend trades.
# Volume confirmation reduces false breakouts. Works in bull/bear via trend alignment.

name = "1d_Camarilla_R3S3_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Camarilla pivot levels
    # Camarilla: Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (Range * 1.1/2) = C + Range * 0.55
    # S3 = C - (Range * 1.1/2) = C - Range * 0.55
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    price_range = df_1w['high'] - df_1w['low']
    camarilla_r3 = typical_price + (price_range * 0.55)
    camarilla_s3 = typical_price - (price_range * 0.55)
    
    # Align Camarilla levels to 1d timeframe (wait for completed 1w bar)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3.values)
    
    # Volume regime: current 1d volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            # Long: break above R3 with volume spike in uptrend
            if close_val > r3 and vol_spike and is_uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short: break below S3 with volume spike in downtrend
            elif close_val < s3 and vol_spike and is_downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        elif position == 1:
            # Long exit: price breaks below S3 OR trend turns down
            if close_val < s3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 OR trend turns up
            if close_val > r3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals