#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw/Teeth/Lips) identifies trend presence and direction.
# 1d EMA50 filter ensures alignment with higher timeframe trend.
# Volume confirmation (>1.8x 20-period average) filters false signals.
# Designed for 4h timeframe targeting 25-40 trades/year to avoid overtrading.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 4h data (13,8,5 periods)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(series, period):
        result = np.full_like(series, np.nan)
        if len(series) >= period:
            sma = np.mean(series[:period])
            result[period-1] = sma
            for i in range(period, len(series)):
                result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price above Jaw + 1d uptrend + volume spike
            if (lips[i] > teeth[i] > jaw[i] and  # bullish alignment
                close[i] > jaw[i] and            # price above Jaw (trend confirmation)
                close[i] > ema_50_1d_aligned[i] and  # price above 1d EMA (higher timeframe uptrend)
                volume[i] > 1.8 * vol_avg_20[i]):  # volume spike
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price below Jaw + 1d downtrend + volume spike
            elif (lips[i] < teeth[i] < jaw[i] and  # bearish alignment
                  close[i] < jaw[i] and            # price below Jaw (trend confirmation)
                  close[i] < ema_50_1d_aligned[i] and  # price below 1d EMA (higher timeframe downtrend)
                  volume[i] > 1.8 * vol_avg_20[i]):  # volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines intertwine (no clear trend) or trend reversal
            if position == 1:
                # Exit long: Alligator loses bullish alignment or trend turns down
                if not (lips[i] > teeth[i] > jaw[i]) or close[i] < ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Alligator loses bearish alignment or trend turns up
                if not (lips[i] < teeth[i] < jaw[i]) or close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0