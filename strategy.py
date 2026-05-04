#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels identify key support/resistance. Breakout above R3 or below S3
# with 1d EMA34 trend alignment and volume confirmation (1.5x 20-period EMA) provides
# high-probability trend continuation entries. Designed for 12h timeframe targeting
# 12-37 trades/year (50-150 total over 4 years) with discrete sizing (0.25).
# Works in bull markets by buying breakouts above R3 in uptrends and in bear markets
# by selling breakdowns below S3 in downtrends, avoiding false breakouts in ranging markets.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    open_price = prices['open'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (based on previous day's OHLC)
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    # S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # We'll use daily high/low/close from 1d data
    camarilla_r3 = df_1d['close'].values + 1.1 * (df_1d['high'].values - df_1d['low'].values)
    camarilla_s3 = df_1d['close'].values - 1.1 * (df_1d['high'].values - df_1d['low'].values)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume confirmation: 1.5x 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: Close breaks above R3 + volume confirmation + price above 1d EMA34 (uptrend)
            if (close[i] > camarilla_r3_aligned[i] and volume_confirmed and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S3 + volume confirmation + price below 1d EMA34 (downtrend)
            elif (close[i] < camarilla_s3_aligned[i] and volume_confirmed and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close falls back below R3 (failed breakout) OR price below 1d EMA34 (trend change)
            if close[i] < camarilla_r3_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close rises back above S3 (failed breakdown) OR price above 1d EMA34 (trend change)
            if close[i] > camarilla_s3_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals