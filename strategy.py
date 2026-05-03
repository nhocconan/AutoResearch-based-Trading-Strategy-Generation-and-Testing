#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level in bull trend (close > 1w EMA50) with volume > 2.0x 20-period MA.
# Short when price breaks below Camarilla S3 level in bear trend (close < 1w EMA50) with volume spike.
# Uses discrete position sizing (0.25) to balance return and drawdown. Camarilla levels from weekly pivot provide
# stronger institutional support/resistance, ideal for 12h timeframe. Volume confirmation ensures institutional participation.
# 1w trend filter reduces whipsaw vs shorter MAs, improving performance in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_R3S3_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for pivot points and EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w pivot points (using previous week's OHLC)
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r3 = prev_high + 2 * (pivot - prev_low)
    s3 = prev_low - 2 * (prev_high - pivot)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align Camarilla levels and EMA to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume regime: current 12h volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
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
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Camarilla breakout conditions
        breakout_up = close_val > r3_aligned[i-1]  # break above previous period's R3
        breakout_down = close_val < s3_aligned[i-1]  # break below previous period's S3
        
        # Entry logic
        if position == 0:
            if is_bull_trend and breakout_up and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and breakout_down and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Camarilla breakout down OR trend reversal
            if breakout_down or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Camarilla breakout up OR trend reversal
            if breakout_up or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals