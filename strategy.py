#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h ADX for trend strength, 6h Williams %R for mean reversion, and volume confirmation.
# Long when ADX > 25 (trending), Williams %R < -80 (oversold), volume > 1.3x average.
# Short when ADX > 25 (trending), Williams %R > -20 (overbought), volume > 1.3x average.
# Exit when ADX < 20 (trend weak) or Williams %R crosses back through -50.
# Designed to capture mean-reversion bounces within strong trends, effective in both bull and bear markets.

name = "6h_ADX_WilliamsR_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ADX trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros(len(high))
        minus_dm = np.zeros(len(high))
        tr = np.zeros(len(high))
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (equivalent to EMA with alpha=1/period)
        atr = np.zeros(len(high))
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = np.zeros(len(high))
        minus_di = np.zeros(len(high))
        dx = np.zeros(len(high))
        
        for i in range(period, len(high)):
            if atr[i] != 0:
                plus_di[i] = 100 * (plus_dm[i] / atr[i])
                minus_di[i] = 100 * (minus_dm[i] / atr[i])
                if (plus_di[i] + minus_di[i]) != 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
        
        adx = np.zeros(len(high))
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Williams %R(14) on 6h data
    def williams_r(high, low, close, period=14):
        highest_high = np.zeros(len(high))
        lowest_low = np.zeros(len(high))
        for i in range(len(high)):
            if i < period:
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                highest_high[i] = np.max(high[i-period+1:i+1])
                lowest_low[i] = np.min(low[i-period+1:i+1])
        
        wr = np.zeros(len(high))
        for i in range(period-1, len(high)):
            if highest_high[i] != lowest_low[i]:
                wr[i] = -100 * (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i])
            else:
                wr[i] = -50
        return wr
    
    wr_6h = williams_r(high, low, close, 14)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_bar = 0
    
    start_idx = 30  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(wr_6h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (strong trend), Williams %R < -80 (oversold), volume spike
            if (adx_12h_aligned[i] > 25 and
                wr_6h[i] < -80 and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
                entry_bar = i
            # Short: ADX > 25 (strong trend), Williams %R > -20 (overbought), volume spike
            elif (adx_12h_aligned[i] > 25 and
                  wr_6h[i] > -20 and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
                entry_bar = i
        elif position == 1:
            # Long exit: ADX < 20 (weak trend) or Williams %R crosses above -50
            if (adx_12h_aligned[i] < 20 or wr_6h[i] > -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: ADX < 20 (weak trend) or Williams %R crosses below -50
            if (adx_12h_aligned[i] < 20 or wr_6h[i] < -50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals