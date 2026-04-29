#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation (>1.5x 20-period average)
# Camarilla pivot levels (R3/S3) act as strong support/resistance; breakouts capture institutional moves.
# 1w EMA34 ensures alignment with weekly trend to avoid counter-trend whipsaws in bear markets like 2022.
# Volume confirmation filters for participation; discrete sizing (0.25) minimizes fee churn.
# Works in bull markets (breakouts with trend) and bear markets (mean reversion off extremes with trend filter).
# Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.

name = "1d_Camarilla_R3S3_1wEMA34_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 1d OHLC for Camarilla pivot levels (using prior day's data)
    # Shift by 1 to avoid look-ahead: use previous day's OHLC for today's levels
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_close = pd.Series(close).shift(1).values
    
    # Camarilla pivot levels: R3, S3 based on prior day's range
    # R3 = close + 1.1*(high-low)*1.1/4 = close + 1.1*(high-low)*0.275
    # S3 = close - 1.1*(high-low)*1.1/4 = close - 1.1*(high-low)*0.275
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 4
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 4
    
    # Calculate 20-period average volume for confirmation (on 1d timeframe)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 1  # 1w EMA34 warmup + shift for prior day OHLC + volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_r3 = camarilla_r3[i]
        curr_s3 = camarilla_s3[i]
        curr_vol_ma = vol_ma_20[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = curr_volume > 1.5 * curr_vol_ma
        
        # Camarilla breakout conditions
        breakout_long = curr_high > curr_r3   # price breaks above R3
        breakout_short = curr_low < curr_s3   # price breaks below S3
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: price retouches mid-point between R3 and S3 OR trend turns bearish
            mid_point = (curr_r3 + curr_s3) / 2
            if curr_close < mid_point or curr_close < curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retouches mid-point between R3 and S3 OR trend turns bullish
            mid_point = (curr_r3 + curr_s3) / 2
            if curr_close > mid_point or curr_close > curr_ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: bullish breakout above R3 AND above 1w EMA34 AND volume confirmation
            if (breakout_long and 
                curr_close > curr_ema_1w and 
                vol_confirm):
                signals[i] = 0.25
                position = 1
            # Short entry: bearish breakout below S3 AND below 1w EMA34 AND volume confirmation
            elif (breakout_short and 
                  curr_close < curr_ema_1w and 
                  vol_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals