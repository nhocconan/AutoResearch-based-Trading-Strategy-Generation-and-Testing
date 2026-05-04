#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels identify key intraday support/resistance. Breakouts above R3 or below S3
# with 1d EMA34 trend alignment and volume spike (2.0x 20-period EMA) provide high-probability
# trend continuation entries. Designed for 12h timeframe to target 12-37 trades/year (50-150 total
# over 4 years) with discrete sizing (0.30). Works in bull markets by buying breakouts in uptrends
# and in bear markets by selling breakdowns in downtrends, avoiding false breakouts in ranging markets.

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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla pivot levels (based on previous day)
    # R4 = close + 1.5*(high-low)*1.1/2, R3 = close + 1.25*(high-low)*1.1/2
    # S3 = close - 1.25*(high-low)*1.1/2, S4 = close - 1.5*(high-low)*1.1/2
    # We need previous day's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_high[1] if len(prev_high) > 1 else high[0]
    prev_low[0] = prev_low[1] if len(prev_low) > 1 else low[0]
    prev_close[0] = prev_close[1] if len(prev_close) > 1 else close[0]
    
    camarilla_range = (prev_high - prev_low) * 1.1
    r3 = prev_close + camarilla_range * 1.25 / 2
    s3 = prev_close - camarilla_range * 1.25 / 2
    
    # Volume confirmation: 2.0x 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: Close breaks above R3 + volume confirmation + price above 1d EMA34 (uptrend)
            if (close[i] > r3[i] and volume_confirmed and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: Close breaks below S3 + volume confirmation + price below 1d EMA34 (downtrend)
            elif (close[i] < s3[i] and volume_confirmed and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: Close falls below R3 (breakout failed) OR price below 1d EMA34 (trend change)
            if close[i] < r3[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: Close rises above S3 (breakdown failed) OR price above 1d EMA34 (trend change)
            if close[i] > s3[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals