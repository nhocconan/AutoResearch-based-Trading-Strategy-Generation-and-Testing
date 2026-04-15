#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Volume-Weighted Average Price (VWAP) deviation with 1d trend filter
# Long when price > VWAP and 1d EMA50 rising; short when price < VWAP and 1d EMA50 falling
# Uses volume confirmation to avoid low-liquidity breakouts
# Target: 20-40 trades/year per symbol to minimize fee drag
# Works in both bull/bear by following higher timeframe trend

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Use expanding window for VWAP (resets daily via alignment)
    vwap_cum = np.nancumsum(vwap_numerator)
    vol_cum = np.nancumsum(vwap_denominator)
    vwap = np.where(vol_cum != 0, vwap_cum / vol_cum, np.nan)
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current > 1.3x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.3 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # Long: price above VWAP + 1d EMA50 rising + volume confirmation
        if (close[i] > vwap[i] and 
            ema_50_aligned[i] > ema_50_aligned[i-1] and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: price below VWAP + 1d EMA50 falling + volume confirmation
        elif (close[i] < vwap[i] and 
              ema_50_aligned[i] < ema_50_aligned[i-1] and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: price crosses VWAP in opposite direction
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < vwap[i]) or
               (signals[i-1] == -0.25 and close[i] > vwap[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_VWAP_EMA50Trend_VolumeFilter"
timeframe = "4h"
leverage = 1.0