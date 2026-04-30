#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Uses 1w HTF for trend filter and 1d Camarilla levels (previous day) to avoid look-ahead.
# Long when price breaks above Camarilla R3 AND price > 1w EMA50 AND volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 AND price < 1w EMA50 AND volume > 2.0x 20-bar average.
# Exit when price crosses Camarilla H3/L3 midline.
# Discrete position sizing (0.25) to limit drawdown and fee churn.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
# Works in bull/bear via 1w EMA50 trend filter and volume confirmation to avoid false breakouts.

name = "1d_Camarilla_R3S3_1wEMA50_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation (shifted by 1 to avoid look-ahead)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: based on previous day's range
    rang = prev_high - prev_low
    camarilla_h3 = prev_close + rang * 1.1 / 6
    camarilla_l3 = prev_close - rang * 1.1 / 6
    camarilla_h3_l3_mid = (camarilla_h3 + camarilla_l3) / 2
    
    # Align Camarilla levels to 1d timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_l3_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3_l3_mid)
    
    # Volume confirmation: volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20)  # warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_h3_l3_mid_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla H3, uptrend (price > 1w EMA50), volume confirmation
            if (curr_high > camarilla_h3_aligned[i] and 
                curr_close > ema_50_1w_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Camarilla S3, downtrend (price < 1w EMA50), volume confirmation
            elif (curr_low < camarilla_l3_aligned[i] and 
                  curr_close < ema_50_1w_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit condition: Camarilla H3/L3 midline cross
            if curr_close < camarilla_h3_l3_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Camarilla H3/L3 midline cross
            if curr_close > camarilla_h3_l3_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals