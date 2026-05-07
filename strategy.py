#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R3 AND 1d close > EMA34 (uptrend) AND volume spike.
# Short when price breaks below S3 AND 1d close < EMA34 (downtrend) AND volume spike.
# Uses Camarilla pivot levels for precise support/resistance, 1d EMA34 for trend direction,
# and volume spike to confirm momentum. Designed for moderate trade frequency (25-40/year).
# Works in bull markets via long breakouts in uptrend and in bear markets via short breakouts in downtrend.
name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla pivot levels for 4h (based on previous bar)
    # Calculate using previous bar's high, low, close
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = pp + (prev_high - prev_low) * 1.1 / 4.0
    s3 = pp - (prev_high - prev_low) * 1.1 / 4.0
    
    # Breakout signals
    breakout_long = (close > r3) & (np.roll(close, 1) <= r3)  # Cross above R3
    breakout_short = (close < s3) & (np.roll(close, 1) >= s3)  # Cross below S3
    
    # Load 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d trend: close > EMA34 (uptrend), close < EMA34 (downtrend)
    trend_up = close_1d > ema_34_1d
    trend_down = close_1d < ema_34_1d
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Breakout above R3 + 1d uptrend + volume spike
            long_condition = breakout_long[i] and trend_up_aligned[i] and volume_spike[i]
            # Short: Breakout below S3 + 1d downtrend + volume spike
            short_condition = breakout_short[i] and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price re-enters below R3 or 1d trend turns down
            if (close[i] < r3[i]) or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters above S3 or 1d trend turns up
            if (close[i] > s3[i]) or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals