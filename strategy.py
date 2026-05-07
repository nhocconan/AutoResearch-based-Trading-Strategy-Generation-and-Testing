# 12h_Camarilla_R3_S3_Breakout_1wTrend_1dVolume
# Hypothesis: Use weekly trend filter (price above/below weekly EMA200) and daily volume confirmation
# to trade Camarilla R3/S3 breakouts on 12h timeframe. Weekly trend ensures we trade with
# higher timeframe momentum, daily volume filter avoids low-liquidity breakouts. Targets 15-25
# trades per year (~60-100 over 4 years) with position size 0.25.
# Weekly trend filter reduces whipsaw in sideways markets, volume confirmation ensures
# institutional participation in breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_1dVolume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Load daily data ONCE for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily volume 20-period average
    vol_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_20_1d)
    
    # Previous period's high, low, close for Camarilla calculation
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    # Calculate Camarilla levels
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    rang = prev_high - prev_low
    r3 = prev_close + rang * 1.1 / 4
    s3 = prev_close - rang * 1.1 / 4
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_20_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Daily volume confirmation: volume > 1.5x 20-day average
        volume_confirm = volume[i] > vol_20_1d_aligned[i] * 1.5
        
        if position == 0:
            # Long: price breaks above R3 in uptrend with volume confirmation
            long_entry = (close[i] > r3[i]) and uptrend and volume_confirm
            # Short: price breaks below S3 in downtrend with volume confirmation
            short_entry = (close[i] < s3[i]) and downtrend and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below S3 or trend changes to downtrend
            if (close[i] < s3[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above R3 or trend changes to uptrend
            if (close[i] > r3[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals