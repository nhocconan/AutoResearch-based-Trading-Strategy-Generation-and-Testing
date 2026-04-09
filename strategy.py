#!/usr/bin/env python3
# 1d_weekly_pivot_vwap_reversion_v1
# Hypothesis: Daily price tends to revert to weekly VWAP (volume-weighted average price),
# with entries triggered when price deviates significantly (>1.5 sigma) from weekly VWAP
# and shows mean-reversion signals (RSI < 30 for long, > 70 for short) during high volume.
# Weekly VWAP acts as dynamic support/resistance; reversion trades work in both bull and bear
# markets as price oscillates around weekly fair value. Uses 1d timeframe for lower frequency
# to minimize fee drag (target: 15-25 trades/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_pivot_vwap_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate weekly VWAP: cumulative (price * volume) / cumulative volume
    # Using typical price = (high + low + close) / 3
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3.0
    pv = typical_price * df_1w['volume']
    cum_pv = pv.cumsum()
    cum_vol = df_1w['volume'].cumsum()
    vwap_1w = (cum_pv / cum_vol).values
    
    # Align weekly VWAP to daily timeframe (completed weekly bars only)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Daily RSI(14) for mean-reversion signals
    delta = pd.Series(close).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup for RSI and volume MA
        # Skip if any required data is NaN
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Calculate deviation from weekly VWAP in percentage
        if vwap_1w_aligned[i] != 0:
            deviation = (close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i]
        else:
            deviation = 0
        
        if position == 1:  # Long position
            # Exit: price returns to weekly VWAP OR RSI exceeds 50 (momentum shift)
            if close[i] >= vwap_1w_aligned[i] or rsi_values[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to weekly VWAP OR RSI falls below 50
            if close[i] <= vwap_1w_aligned[i] or rsi_values[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Volume confirmation
            volume_confirmed = volume[i] > 1.5 * volume_ma[i]
            
            if volume_confirmed:
                # Long signal: price significantly below VWAP + oversold RSI
                if deviation < -0.015 and rsi_values[i] < 30:  # >1.5% below VWAP + RSI < 30
                    position = 1
                    signals[i] = 0.25
                # Short signal: price significantly above VWAP + overbought RSI
                elif deviation > 0.015 and rsi_values[i] > 70:  # >1.5% above VWAP + RSI > 70
                    position = -1
                    signals[i] = -0.25
    
    return signals