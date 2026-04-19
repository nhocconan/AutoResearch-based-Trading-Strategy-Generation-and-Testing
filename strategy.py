#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h volume-weighted VWAP deviation + 1d trend filter + volume spike
# Price deviation from VWAP indicates short-term mean reversion opportunities
# Combined with daily trend filter to trade in direction of higher timeframe
# Volume spike confirms institutional participation
# Target: 20-30 trades/year per symbol with disciplined entries
name = "4h_VWAPDeviation_1dTrend_Volume"
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
    
    # Daily EMA50 for trend bias
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # VWAP calculation (volume weighted average price)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative VWAP with reset each day (using 4h bars)
    # Approximate daily reset by using 6-period lookback (24h/4h = 6)
    cum_vwap_num = pd.Series(vwap_numerator).rolling(window=6, min_periods=1).sum().values
    cum_vwap_den = pd.Series(vwap_denominator).rolling(window=6, min_periods=1).sum().values
    vwap = np.where(cum_vwap_den != 0, cum_vwap_num / cum_vwap_den, typical_price)
    
    # Price deviation from VWAP (%)
    vwap_deviation = (close - vwap) / vwap * 100
    
    # Volume spike: volume > 2.0 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vwap_deviation[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price below VWAP (oversold) + above daily EMA + volume spike
            if (vwap_deviation[i] < -1.0 and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price above VWAP (overbought) + below daily EMA + volume spike
            elif (vwap_deviation[i] > 1.0 and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns to VWAP or breaks below daily EMA
            if (vwap_deviation[i] > 0.0) or (close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns to VWAP or breaks above daily EMA
            if (vwap_deviation[i] < 0.0) or (close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals