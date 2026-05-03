#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA(34) trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3, price > 1w EMA34, and volume > 2.0x 20-bar average
# Short when price breaks below Camarilla S3, price < 1w EMA34, and volume > 2.0x 20-bar average
# Uses 1d timeframe for lower trade frequency (target: 30-100 total trades over 4 years)
# 1w EMA provides higher timeframe trend alignment to avoid counter-trend trades
# Volume spike confirms breakout strength
# Designed for BTC/ETH with discrete position sizing (0.25) to minimize fee drag
# Works in bull (breakouts above rising EMA) and bear (breakdowns below falling EWA) markets

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA(34) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for completed 1w bar)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels on 1d (using previous day's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.25*(high-low), 
    #            S3 = close - 1.25*(high-low), S4 = close - 1.5*(high-low)
    # We use R3 and S3 for breakout entries
    prev_close = prices['close'].shift(1).values
    prev_high = prices['high'].shift(1).values
    prev_low = prices['low'].shift(1).values
    camarilla_r3 = prev_close + 1.25 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.25 * (prev_high - prev_low)
    
    # Volume confirmation (2.0x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(34, 20) + 1  # EMA(34) + volume MA(20) warmup + shift(1)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price > Camarilla R3, price > 1w EMA34, volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price < Camarilla S3, price < 1w EMA34, volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price < Camarilla S3 or price < 1w EMA34
            if (close[i] < camarilla_s3[i] or 
                close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price > Camarilla R3 or price > 1w EMA34
            if (close[i] > camarilla_r3[i] or 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals