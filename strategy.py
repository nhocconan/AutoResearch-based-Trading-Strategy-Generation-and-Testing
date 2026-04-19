#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot reversal from R1/S1 with 12h EMA34 filter and volume confirmation.
# Long when price crosses below S1, closes above S1, and price > 12h EMA34 and volume > 1.5x 6h average volume.
# Short when price crosses above R1, closes below R1, and price < 12h EMA34 and volume > 1.5x 6h average volume.
# Exit when price crosses the 12h EMA34 or when price reaches R3/S3 (profit target).
# Uses Camarilla for reversal zones, EMA for trend filter, volume for confirmation.
# Target: 15-30 trades/year per symbol to stay within frequency limits.
name = "6h_Camarilla_R1_S1_Reversal_EMA34_Volume"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R3 = prev_close + 1.1 * prev_range * 1.1 / 12
    R2 = prev_close + 1.1 * prev_range * 1.1 / 6
    R1 = prev_close + 1.1 * prev_range * 1.1 / 4
    S1 = prev_close - 1.1 * prev_range * 1.1 / 4
    S2 = prev_close - 1.1 * prev_range * 1.1 / 6
    S3 = prev_close - 1.1 * prev_range * 1.1 / 12
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Get 12h data for EMA34 filter
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 6h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure EMA34 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        ema_34 = ema_34_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long entry: price crosses below S1, closes above S1, and above EMA34 with volume confirmation
            if i > 0 and low[i-1] < s1_aligned[i-1] and close[i-1] < s1_aligned[i-1] and \
               price > s1 and price > ema_34 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short entry: price crosses above R1, closes below R1, and below EMA34 with volume confirmation
            elif i > 0 and high[i-1] > r1_aligned[i-1] and close[i-1] > r1_aligned[i-1] and \
                 price < r1 and price < ema_34 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA34 or reaches S3 (stop)
            if price < ema_34 or price < s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA34 or reaches R3 (stop)
            if price > ema_34 or price > r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals