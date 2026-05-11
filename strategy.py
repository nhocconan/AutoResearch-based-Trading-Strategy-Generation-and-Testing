#/usr/bin/env python3
name = "1d_Camarilla_R3S3_Breakout_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # 1. Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 2. Weekly EMA50 for trend filter (weekly close EMA)
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 3. Load daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 4. Daily ATR for volatility filter
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 5. Calculate Camarilla levels: R3, S3
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.11
    s3 = close_1d - hl_range * 1.11
    
    # 6. Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 7. Volume filter: 20-period EMA for higher threshold
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 8. Fixed position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_weekly_ema = close[i] > ema50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema50_1w_aligned[i]
        breakout_long = close[i] > r3[i]
        breakout_short = close[i] < s3[i]
        volatility_ok = atr[i] > np.nanpercentile(atr[max(0, i-50):i+1], 20) if i >= 50 else True
        
        if position == 0:
            # Long: Price breaks above R3 + above weekly EMA50 + volume spike + volatility filter
            if breakout_long and price_above_weekly_ema and volume_ok[i] and volatility_ok:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below weekly EMA50 + volume spike + volatility filter
            elif breakout_short and price_below_weekly_ema and volume_ok[i] and volatility_ok:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses
                if close[i] < s3[i] or close[i] < ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses
                if close[i] > r3[i] or close[i] > ema50_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals