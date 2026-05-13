#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above R3 with 4h EMA50 uptrend and volume > 1.5x 20-bar average.
# Short when price breaks below S3 with 4h EMA50 downtrend and volume > 1.5x 20-bar average.
# Uses session filter (08-20 UTC) to avoid low-liquidity hours.
# Target: 15-35 trades/year per symbol by using tight 1h entry timing with 4h trend alignment.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 20-bar volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h close
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla pivot levels for 1h using previous bar's OHLC
    # Camarilla: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # where C = (H+L+O)/3 (typical price)
    # We need previous bar's OHLC to avoid look-ahead
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_open = np.roll(prices['open'].values, 1)
    
    # First bar has no previous data
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    prev_open[0] = prices['open'].iloc[0]
    
    # Calculate typical price (pivot point)
    pp = (prev_high + prev_low + prev_close) / 3.0
    # Calculate Camarilla R3 and S3
    r3 = pp + (prev_high - prev_low) * 1.1 / 2.0
    s3 = pp - (prev_high - prev_low) * 1.1 / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for volume MA
        # Skip if any required data is NaN or not in session
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above R3, 4h EMA50 uptrend (close > EMA50), volume spike
            if (close[i] > r3[i] and 
                close_4h[i // 16] > ema50_4h[i // 16] if i // 16 < len(ema50_4h) else False and  # Simplified 4h trend check
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: price breaks below S3, 4h EMA50 downtrend (close < EMA50), volume spike
            elif (close[i] < s3[i] and 
                  close_4h[i // 16] < ema50_4h[i // 16] if i // 16 < len(ema50_4h) else False and  # Simplified 4h trend check
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below S3 (reversal signal)
            if close[i] < s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: price breaks above R3 (reversal signal)
            if close[i] > r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals