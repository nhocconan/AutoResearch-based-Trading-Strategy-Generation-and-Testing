#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 level in bull trend (close > 4h EMA50) with volume > 1.8x 20-period MA.
# Short when price breaks below Camarilla S3 level in bear trend (close < 4h EMA50) with volume spike.
# Uses discrete position sizing (0.20) to minimize fee churn while maintaining sufficient exposure.
# 4h EMA50 provides higher timeframe trend filter to avoid counter-trend trades in both bull and bear markets.
# Volume confirmation ensures moves have institutional participation, reducing false signals.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Target: 80-120 total trades over 4 years (20-30/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError with datetime64
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h data for EMA50 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels on 4h data (using previous 4h bar's OHLC)
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C = close, H = high, L = low of previous 4h bar
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    camarilla_r3 = prev_close_4h + (prev_high_4h - prev_low_4h) * 1.1 / 2
    camarilla_s3 = prev_close_4h - (prev_high_4h - prev_low_4h) * 1.1 / 2
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume regime: current 1h volume > 1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_4h_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if in_session and is_bull_trend and close_val > r3_level and vol_spike:
                signals[i] = 0.20
                position = 1
            elif in_session and is_bear_trend and close_val < s3_level and vol_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price below S3 level OR trend reversal
            if close_val < s3_level or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price above R3 level OR trend reversal
            if close_val > r3_level or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals