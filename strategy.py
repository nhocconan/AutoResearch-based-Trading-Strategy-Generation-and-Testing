#!/usr/bin/env python3
"""
4h_BackwardationBreakout
Hypothesis: Use daily backwardation (spot price below 24h VWAP) to identify oversold conditions,
enter long when price breaks above Donchian(20) high with volume confirmation,
exit on reversal below Donchian(20) low or RSI(14) > 70.
Works in bull markets by catching breakouts from pullbacks and in bear markets by
fading breakdowns with mean-reversion bias. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 24:
        return np.zeros(n)
    
    # === Daily 24h VWAP for backwardation filter ===
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.values
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # === Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === RSI(14) for exit filter ===
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nanmean(arr[1:period]) if period > 1 else arr[0]
        for i in range(period, len(arr)):
            if np.isnan(result[i-1]) or np.isnan(arr[i]):
                result[i] = np.nan
            else:
                result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    period_rsi = 14
    avg_gain = wilder_smooth(gain, period_rsi)
    avg_loss = wilder_smooth(loss, period_rsi)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan], rsi])
    
    # === Volume confirmation ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if indicators not ready
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        vwap = vwap_1d_aligned[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Long: backwardation + Donchian breakout + volume
            if (price_close < vwap and  # backwardation condition
                price_close > upper and
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: contango + breakdown + volume
            elif (price_close > vwap and  # contango condition
                  price_close < lower and
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            if position == 1:
                if price_close < lower or rsi_val > 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > upper or rsi_val < 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_BackwardationBreakout"
timeframe = "4h"
leverage = 1.0