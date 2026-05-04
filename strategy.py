#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above Camarilla R3 level AND 1d close > EMA34 AND volume > 2.0x 20-period volume EMA
# Short when price breaks below Camarilla S3 level AND 1d close < EMA34 AND volume > 2.0x 20-period volume EMA
# Exit on opposite Camarilla level touch (S3 for longs, R3 for shorts) OR trend reversal
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 25-40 trades/year per symbol.
# Camarilla levels provide intraday structure proven effective in ranging/volatile markets.
# Volume spike (2.0x) and 1d EMA34 trend filter reduce false breakouts.
# Works in bull markets via longs in uptrend and bear markets via shorts in downtrend.

name = "4h_Camarilla_R3S3_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_1d = close_1d > ema_34_1d
    trend_bearish_1d = close_1d < ema_34_1d
    
    # Align 1d trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1d, trend_bullish_1d.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1d, trend_bearish_1d.astype(float))
    
    # Calculate Camarilla levels for 4h timeframe using previous bar's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low)
    #          S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    # We use previous bar to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    rang = prev_high - prev_low
    camarilla_r3 = prev_close + 1.125 * rang
    camarilla_s3 = prev_close - 1.125 * rang
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2.0x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d bullish trend AND volume spike
            if (close[i] > camarilla_r3[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1d bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d bearish trend AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1d bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Camarilla S3 OR 1d trend turns bearish
            if (close[i] <= camarilla_s3[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Camarilla R3 OR 1d trend turns bullish
            if (close[i] >= camarilla_r3[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals