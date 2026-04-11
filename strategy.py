# #!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with weekly trend filter and daily RSI extremes.
# Uses weekly EMA(34) to determine trend direction and daily RSI(14) for entry.
# Enters long when weekly trend is up and daily RSI < 30 (oversold).
# Enters short when weekly trend is down and daily RSI > 70 (overbought).
# Volume filter requires current volume > 1.3 * 20-period average volume.
# Designed for 12-37 trades/year with clear trend-following logic.

name = "12h_1w1d_rsi_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly and daily data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 35 or len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend direction
    close_1w = df_1w['close'].values
    ema_34_1w = np.full_like(close_1w, np.nan, dtype=float)
    alpha = 2 / (34 + 1)
    for i in range(len(close_1w)):
        if i == 0:
            ema_34_1w[i] = close_1w[i]
        elif np.isnan(ema_34_1w[i-1]):
            ema_34_1w[i] = close_1w[i]
        else:
            ema_34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema_34_1w[i-1]
    
    # Weekly trend: price above EMA = bullish, below EMA = bearish
    weekly_trend_up = close_1w > ema_34_1w
    weekly_trend_down = close_1w < ema_34_1w
    
    # Align weekly trend to 12h
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Calculate daily RSI(14) for entry signals
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/14)
    alpha_rsi = 1 / 14
    avg_gain = np.full_like(gain, np.nan, dtype=float)
    avg_loss = np.full_like(loss, np.nan, dtype=float)
    
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * (i-1) + gain[i]) / i if i > 0 else gain[i]
                avg_loss[i] = (avg_loss[i-1] * (i-1) + loss[i]) / i if i > 0 else loss[i]
        else:
            avg_gain[i] = alpha_rsi * gain[i] + (1 - alpha_rsi) * avg_gain[i-1]
            avg_loss[i] = alpha_rsi * loss[i] + (1 - alpha_rsi) * avg_loss[i-1]
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_avg_20[i] = np.mean(volume_1d[i-19:i+1])
    
    # Align daily indicators to 12h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if any required data is invalid
        if (np.isnan(rsi_aligned[i]) or np.isnan(vol_avg_aligned[i]) or
            np.isnan(weekly_trend_up_aligned[i]) or np.isnan(weekly_trend_down_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.3 * daily average volume
        vol_filter = volume[i] > 1.3 * vol_avg_aligned[i]
        
        # Determine weekly trend direction
        is_up_trend = weekly_trend_up_aligned[i]
        is_down_trend = weekly_trend_down_aligned[i]
        
        # Entry conditions
        rsi_value = rsi_aligned[i]
        
        # Long: weekly trend up + RSI oversold (<30) + volume
        long_signal = is_up_trend and (rsi_value < 30) and vol_filter
        # Short: weekly trend down + RSI overbought (>70) + volume
        short_signal = is_down_trend and (rsi_value > 70) and vol_filter
        
        # Exit conditions: opposite RSI extreme or trend change
        exit_long = (position == 1 and 
                    (rsi_value > 70 or not is_up_trend))
        exit_short = (position == -1 and 
                     (rsi_value < 30 or not is_down_trend))
        
        # Update position and signals
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals