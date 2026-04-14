#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Volume Weighted Average Price (VWAP) as dynamic support/resistance
# with 1-week RSI trend filter and volume confirmation. VWAP acts as institutional reference point
# where price tends to revert or break with institutional flow. Long when price crosses above VWAP
# in uptrend (1w RSI > 50) with volume confirmation, short when below VWAP in downtrend (1w RSI < 50).
# Exit when price crosses back across VWAP or RSI flips. Designed to work in both bull/bear markets
# by using VWAP as dynamic equilibrium and RSI for trend filter. Target: 15-25 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate typical price and VWAP for each 1d bar
    typical_price_1d = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = typical_price_1d * volume_1d
    vwap_denominator = volume_1d
    
    # Cumulative VWAP (resets daily)
    cum_vwap_num = np.cumsum(vwap_numerator)
    cum_vwap_den = np.cumsum(vwap_denominator)
    vwap_1d = np.divide(cum_vwap_num, cum_vwap_den, out=np.full_like(cum_vwap_num, np.nan), where=cum_vwap_den!=0)
    
    # Load 1w data ONCE for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI(14) on 1w
    delta = np.diff(close_1w, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align indicators to lower timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume confirmation: 1.3x average volume (slightly lower to avoid whipsaws)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # Need volume MA and RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: RSI > 50 for uptrend, < 50 for downtrend
        uptrend = rsi_1w_aligned[i] > 50
        downtrend = rsi_1w_aligned[i] < 50
        
        if position == 0:
            # Look for VWAP crossovers with confirmation
            # Long: price crosses above VWAP AND uptrend AND volume
            if (close[i] > vwap_1d_aligned[i] and 
                close[i-1] <= vwap_1d_aligned[i-1] and  # crossed above this bar
                uptrend and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price crosses below VWAP AND downtrend AND volume
            elif (close[i] < vwap_1d_aligned[i] and 
                  close[i-1] >= vwap_1d_aligned[i-1] and  # crossed below this bar
                  downtrend and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below VWAP or RSI flips to downtrend
            if (close[i] < vwap_1d_aligned[i] and 
                close[i-1] >= vwap_1d_aligned[i-1]) or \
               rsi_1w_aligned[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back above VWAP or RSI flips to uptrend
            if (close[i] > vwap_1d_aligned[i] and 
                close[i-1] <= vwap_1d_aligned[i-1]) or \
               rsi_1w_aligned[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_VWAP_1wRSI_Volume_v1"
timeframe = "6h"
leverage = 1.0