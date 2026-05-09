# 6H_CAMARILLA_R3_S3_BREAKOUT_12HTREND_VOLUME_CONFIRMED
# Hypothesis: Camarilla R3/S3 breakouts on 6h with 12h EMA50 trend filter and volume spike confirmation
# Works in bull markets via breakout continuation, in bear markets via faded rejections at R3/S3
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6H_CAMARILLA_R3_S3_BREAKOUT_12HTREND_VOLUME_CONFIRMED"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = pd.Series(df_12h['close'].values)
    ema50_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    # Typical price = (H+L+C)/3
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    range_12h = df_12h['high'] - df_12h['low']
    
    # Camarilla levels
    r3 = typical_price + 1.1 * range_12h * 1.1 / 2
    s3 = typical_price - 1.1 * range_12h * 1.1 / 2
    r4 = typical_price + 1.1 * range_12h * 1.1
    s4 = typical_price - 1.1 * range_12h * 1.1
    
    # Shift levels by 1 to avoid look-ahead (use previous bar's levels)
    r3 = r3.shift(1)
    s3 = s3.shift(1)
    r4 = r4.shift(1)
    s4 = s4.shift(1)
    
    # Align to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_12h, r3.values)
    s3_6h = align_htf_to_ltf(prices, df_12h, s3.values)
    r4_6h = align_htf_to_ltf(prices, df_12h, r4.values)
    s4_6h = align_htf_to_ltf(prices, df_12h, s4.values)
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Break above R3 with volume and above 12h EMA50 trend
            if close[i] > r3_6h[i] and vol_ok and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume and below 12h EMA50 trend
            elif close[i] < s3_6h[i] and vol_ok and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below R3 or trend reversal
            if close[i] < r3_6h[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above S3 or trend reversal
            if close[i] > s3_6h[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals