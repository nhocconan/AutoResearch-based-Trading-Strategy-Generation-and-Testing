#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h CCI mean reversion with 1-day ADX filter and volume spike confirmation.
# Long when: CCI(20) crosses below -100 (oversold), ADX(14) < 20 (range), volume > 1.5x 20-period average
# Short when: CCI(20) crosses above +100 (overbought), ADX(14) < 20 (range), volume > 1.5x 20-period average
# Exit when CCI returns to opposite side of zero (long exits at CCI>0, short exits at CCI<0)
# CCI captures overbought/oversold extremes in ranging markets, ADX filters for low volatility regimes,
# volume spike confirms reversal conviction. Target: 15-25 trades/year per symbol.
# Works in bull (buy oversold dips in ranges) and bear (sell overbought rallies in ranges).
name = "12h_CCI_ADX_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smooth = np.sum(plus_dm[1:period+1])
        minus_dm_smooth = np.sum(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smooth = plus_dm_smooth - (plus_dm_smooth / period) + plus_dm[i]
            minus_dm_smooth = minus_dm_smooth - (minus_dm_smooth / period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_smooth / (atr[i] * period) if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_smooth / (atr[i] * period) if atr[i] != 0 else 0
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0
            if i >= 2*period:
                adx[i] = np.mean(dx[i-period+1:i+1])
        
        return adx
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate CCI(20) on 12h data
    typical_price = (high + low + close) / 3.0
    sma_tp = pd.Series(typical_price).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(typical_price).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (typical_price - sma_tp) / (0.015 * mad)
    cci = np.where(mad == 0, 0, cci)
    
    # 20-period volume average for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(cci[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        cci_val = cci[i]
        adx_val = adx_14_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: CCI crosses below -100 from above, ADX < 20 (range), volume spike
            if (cci_val < -100 and cci[i-1] >= -100 and 
                adx_val < 20 and vol > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short entry: CCI crosses above +100 from below, ADX < 20 (range), volume spike
            elif (cci_val > 100 and cci[i-1] <= 100 and 
                  adx_val < 20 and vol > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI returns above zero (mean reversion)
            if cci_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI returns below zero (mean reversion)
            if cci_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals