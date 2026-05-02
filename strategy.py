#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 1d EMA50 trend filter
# Williams Alligator (jaw=13, teeth=8, lips=5) identifies trend absence (alligator sleeping)
# Elder Ray measures bull/bear power via EMA13: bull_power = high - EMA13, bear_power = EMA13 - low
# Long when: alligator awake (jaws > teeth > lips) AND bull_power > 0 AND price > 1d EMA50
# Short when: alligator awake (jaws < teeth < lips) AND bear_power > 0 AND price < 1d EMA50
# Uses 6h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# 1d EMA50 ensures alignment with daily trend to avoid counter-trend entries
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure
# Works in bull (continuation) and bear (mean reversion via short) markets by following the alligator's trend

name = "6h_WilliamsAlligator_ElderRay_1dEMA50_Trend_Volume_v1"
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
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator: SMAs of median price (hlc3)
    hlc3 = (high + low + close) / 3
    jaw = pd.Series(hlc3).rolling(window=13, min_periods=13).mean().values  # slowest
    teeth = pd.Series(hlc3).rolling(window=8, min_periods=8).mean().values    # medium
    lips = pd.Series(hlc3).rolling(window=5, min_periods=5).mean().values    # fastest
    
    # Elder Ray: EMA13 of close for bull/bear power calculation
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # bull power: high minus EMA13
    bear_power = ema13 - low   # bear power: EMA13 minus low
    
    # Volume confirmation: volume > 1.5 * 20-period EMA (6h)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine alligator state
        alligator_long = jaw[i] > teeth[i] and teeth[i] > lips[i]  # jaws > teeth > lips (uptrend)
        alligator_short = jaw[i] < teeth[i] and teeth[i] < lips[i]  # jaws < teeth < lips (downtrend)
        
        # Determine trend bias from 1d EMA50
        bullish_bias = close[i] > ema_50_1d_aligned[i]
        bearish_bias = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if alligator_long and bullish_bias and bull_power[i] > 0 and volume_spike[i]:
                # Long: alligator awake uptrend + daily trend bullish + bull power + volume
                signals[i] = 0.25
                position = 1
            elif alligator_short and bearish_bias and bear_power[i] > 0 and volume_spike[i]:
                # Short: alligator awake downtrend + daily trend bearish + bear power + volume
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0  # No clear signal
        
        elif position == 1:  # Long position
            # Exit: alligator sleeping (jaws < teeth) OR price below 1d EMA50 OR bear power > bull power
            if jaw[i] < teeth[i] or close[i] < ema_50_1d_aligned[i] or bear_power[i] > bull_power[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: alligator sleeping (jaws > teeth) OR price above 1d EMA50 OR bull power > bear power
            if jaw[i] > teeth[i] or close[i] > ema_50_1d_aligned[i] or bull_power[i] > bear_power[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals