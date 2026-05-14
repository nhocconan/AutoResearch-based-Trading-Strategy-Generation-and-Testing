#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 12h trend filter and volume confirmation.
# Long when Williams %R crosses above -20 (oversold reversal) AND 12h close > EMA50 (uptrend) AND volume spike.
# Short when Williams %R crosses below -80 (overbought reversal) AND 12h close < EMA50 (downtrend) AND volume spike.
# Uses Williams %R for mean reversion in extremes, 12h EMA50 for trend direction, and volume to confirm momentum.
# Designed for moderate trade frequency (target: 20-40/year) to balance opportunity and cost.
# Works in bull markets via long reversals in uptrend and in bear markets via short reversals in downtrend.
name = "6h_WilliamsR_Reversal_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period) on 6h data
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0, williams_r, -50.0)
    
    # Williams %R signals: cross above -20 (long), cross below -80 (short)
    williams_r_long_signal = (williams_r > -20) & (np.roll(williams_r, 1) <= -20)
    williams_r_short_signal = (williams_r < -80) & (np.roll(williams_r, 1) >= -80)
    
    # Load 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h trend: close > EMA50 (uptrend), close < EMA50 (downtrend)
    trend_up = close_12h > ema_50_12h
    trend_down = close_12h < ema_50_12h
    trend_up_aligned = align_htf_to_ltf(prices, df_12h, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_12h, trend_down)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for Williams %R and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R oversold reversal + 12h uptrend + volume spike
            long_condition = williams_r_long_signal[i] and trend_up_aligned[i] and volume_spike[i]
            # Short: Williams %R overbought reversal + 12h downtrend + volume spike
            short_condition = williams_r_short_signal[i] and trend_down_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Williams %R crosses below -50 (momentum loss) or 12h trend turns down
            if (williams_r[i] < -50 and np.roll(williams_r, 1)[i] >= -50) or not trend_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Williams %R crosses above -50 (momentum loss) or 12h trend turns up
            if (williams_r[i] > -50 and np.roll(williams_r, 1)[i] <= -50) or not trend_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals