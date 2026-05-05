#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 12h Camarilla R3 level AND 1d close > 1d EMA34 AND volume > 2.0x 20-period average
# Short when price breaks below 12h Camarilla S3 level AND 1d close < 1d EMA34 AND volume > 2.0x 20-period average
# Exit when price crosses 1d EMA34 (trend reversal) OR price returns to Camarilla pivot point (mean reversion)
# Uses 12h primary timeframe with 1d HTF for trend filter
# Camarilla levels provide precise intraday support/resistance with statistical edge
# Volume confirmation reduces false breakouts
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h Camarilla levels (based on previous 12h bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low)*1.1/8, R3 = close + 1.1*(high-low)*1.1/8, etc.
    # We need previous bar's OHLC to calculate current bar's levels
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # First bar has no previous
    
    range_hl = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3
    camarilla_r3 = camarilla_pivot + 1.1 * range_hl * 1.1 / 8
    camarilla_s3 = camarilla_pivot - 1.1 * range_hl * 1.1 / 8
    
    # Volume confirmation: volume > 2.0x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or 
            np.isnan(camarilla_pivot[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND 1d close > 1d EMA34 AND volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND 1d close < 1d EMA34 AND volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 1d EMA34 (trend reversal) OR returns to pivot point (mean reversion)
            if close[i] < ema_34_1d_aligned[i] or close[i] < camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 1d EMA34 (trend reversal) OR returns to pivot point (mean reversion)
            if close[i] > ema_34_1d_aligned[i] or close[i] > camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals