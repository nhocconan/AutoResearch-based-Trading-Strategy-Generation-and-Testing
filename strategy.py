#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R3/S3 breakouts with 1-week EMA34 trend filter and volume confirmation.
Targets 30-100 total trades over 4 years by requiring confluence of weekly trend, volume spike, and Camarilla breakout.
Works in bull/bear markets via weekly trend filter (EMA34). Camarilla R3/S3 levels provide institutional support/resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivot levels on daily timeframe (requires previous day's OHLC)
    # Camarilla levels based on previous day's range
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_open = np.roll(open_, 1)
    
    # First day will have NaN due to roll
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_open[0] = np.nan
    
    # Calculate daily range and Camarilla levels
    range_ = prev_high - prev_low
    camarilla_r3 = prev_close + (range_ * 1.1 / 4)
    camarilla_s3 = prev_close - (range_ * 1.1 / 4)
    camarilla_r4 = prev_close + (range_ * 1.1 / 2)
    camarilla_s4 = prev_close - (range_ * 1.1 / 2)
    
    # Volume spike: volume > 2.0x 20-period median volume
    volume_series = pd.Series(volume)
    vol_median_20 = volume_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (2.0 * vol_median_20)
    
    # Fixed position size to control trade frequency and drawdown
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 34 for 1w EMA, 1 for previous day data, 20 for volume median
    start_idx = max(34, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or
            np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_34_val = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = fixed_size
        
        if position == 0:
            # Flat - look for entry
            # Bullish breakout: close > Camarilla R3 and volume spike, in uptrend (close > EMA34_1w)
            long_entry = (close_val > camarilla_r3[i]) and vol_spike and (close_val > ema_34_val)
            # Bearish breakout: close < Camarilla S3 and volume spike, in downtrend (close < EMA34_1w)
            short_entry = (close_val < camarilla_s3[i]) and vol_spike and (close_val < ema_34_val)
            
            if long_entry:
                signals[i] = size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal or at Camarilla R4 (take profit)
            if close_val < ema_34_val or close_val > camarilla_r4[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on trend reversal or at Camarilla S4 (take profit)
            if close_val > ema_34_val or close_val < camarilla_s4[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0