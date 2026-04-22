#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and 1w volume confirmation
# Williams Alligator uses 3 smoothed SMAs (Jaw: 13, Teeth: 8, Lips: 5) to detect trends.
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish alignment).
# Trend filter: 1d EMA34 (price above = bullish bias, below = bearish bias).
# Volume confirmation: 1w volume > 1.5x 4-week average to avoid false signals.
# Works in bull markets by capturing strong uptrends and in bear markets by avoiding false signals
# via trend filter and requiring volume confirmation. Designed for 12h timeframe targeting 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Load 1w data for volume confirmation (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    
    # Williams Alligator on 12h data: 3 smoothed SMAs
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan)
        sma = pd.Series(series).rolling(window=period, min_periods=period).mean().values
        smma_vals = np.full_like(series, np.nan)
        smma_vals[period-1] = sma[period-1]
        for i in range(period, len(series)):
            smma_vals[i] = (smma_vals[i-1] * (period-1) + series[i]) / period
        return smma_vals
    
    jaw = smma(high, 13)  # Using high for Jaw as per original Alligator
    teeth = smma(low, 8)   # Using low for Teeth
    lips = smma(close, 5)  # Using close for Lips
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1w volume 4-period average for spike detection
    vol_avg_4_1w = pd.Series(volume_1w).rolling(window=4, min_periods=4).mean().values
    vol_avg_4_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_4_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_4_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + 1d uptrend + 1w volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * vol_avg_4_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + 1d downtrend + 1w volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_4_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Alligator lines cross or trend reversal
            if position == 1:
                # Exit on bearish cross (Lips < Teeth) or trend reversal
                if (lips[i] < teeth[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit on bullish cross (Lips > Teeth) or trend reversal
                if (lips[i] > teeth[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_1wVolSpike"
timeframe = "12h"
leverage = 1.0