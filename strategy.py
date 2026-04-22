#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation
# Williams Alligator identifies trend direction via smoothed moving averages (Jaw/Teeth/Lips).
# Long when Lips > Teeth > Jaw (bullish alignment), short when Lips < Teeth < Jaw (bearish).
# Entry confirmed by 1d EMA34 trend and 1d volume spike (>1.5x 20-day average) to avoid false signals.
# Works in bull markets by capturing strong uptrends and in bear markets by avoiding false reversals
# via trend filter and volume confirmation. Designed for 12h timeframe targeting 12-37 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and volume confirmation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator on 12h data: SMMA(13,8), SMMA(8,5), SMMA(5,3)
    def smoothed_moving_average(arr, period):
        sma = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return sma
        sma[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            sma[i] = (sma[i-1] * (period-1) + arr[i]) / period
        return sma
    
    jaw = smoothed_moving_average(close, 13)  # SMMA(13,8) -> 8-period smoothing
    teeth = smoothed_moving_average(close, 8)  # SMMA(8,5) -> 5-period smoothing
    lips = smoothed_moving_average(close, 5)   # SMMA(5,3) -> 3-period smoothing
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume 20-period average for spike detection
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and
                close[i] > ema_34_1d_aligned[i] and
                volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and
                  close[i] < ema_34_1d_aligned[i] and
                  volume[i] > 1.5 * vol_avg_20_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross or trend reversal
            if position == 1:
                # Exit on bearish cross (Lips < Teeth or Teeth < Jaw) or trend reversal
                if (lips[i] < teeth[i] or teeth[i] < jaw[i] or
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish cross (Lips > Teeth or Teeth > Jaw) or trend reversal
                if (lips[i] > teeth[i] or teeth[i] > jaw[i] or
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0