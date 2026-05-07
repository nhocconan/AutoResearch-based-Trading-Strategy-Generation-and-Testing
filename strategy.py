#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation.
# Long when price breaks above R3 and 12h close > EMA34 (uptrend) and volume spike.
# Short when price breaks below S3 and 12h close < EMA34 (downtrend) and volume spike.
# Uses Camarilla levels for institutional support/resistance, 12h EMA34 for trend direction, and volume to confirm momentum.
# Designed for low trade frequency (target: 20-30/year) to minimize fee drag and maximize edge.
# Works in bull markets via long breakouts in uptrend and in bear markets via short breakouts in downtrend.
name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Camarilla levels from previous day (requires daily high/low/close)
    # We'll use 12h data to approximate daily range: each day = 2 bars of 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 40:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Approximate daily OHLC from 12h bars: group every 2 bars
    # We'll compute Camarilla for each 12h bar using the prior 2-bar session (1 day)
    # To avoid look-ahead, we use the completed session before the current 12h bar
    # For simplicity, we'll use the prior 12h bar's high/low/close as proxy for prior day
    # This is acceptable as 12h bar represents half a day, and we only need range
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    # Avoid division by zero
    range_safe = np.where(range_ == 0, 1e-10, range_)
    
    # R3 and S3 levels
    R3 = prev_close + range_ * 1.1 / 4
    S3 = prev_close - range_ * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_12h, R3)
    S3_aligned = align_htf_to_ltf(prices, df_12h, S3)
    
    # 12h trend filter: EMA34
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up = close_12h > ema_34_12h
    trend_down = close_12h < ema_34_12h
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume confirmation: current volume > 2.0 * 20-period EMA (higher threshold for fewer trades)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Sufficient warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + 12h uptrend + volume spike
            long_condition = close[i] > R3_aligned[i] and trend_up_aligned[i] and volume_spike[i]
            # Short: price breaks below S3 + 12h downtrend + volume spike
            short_condition = close[i] < S3_aligned[i] and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S3 (reversal) or 12h trend turns down
            if close[i] < S3_aligned[i] or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R3 (reversal) or 12h trend turns up
            if close[i] > R3_aligned[i] or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals