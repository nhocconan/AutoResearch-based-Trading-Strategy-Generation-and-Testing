Allocation: 0.25

# Hypothesis: In a bearish/ranging market (2025+), price tends to revert to the weekly VWAP after extreme deviations. 
# This strategy goes long when price deviates significantly below weekly VWAP with confirmation from daily RSI oversold and volume spike,
# and short when price deviates significantly above weekly VWAP with daily RSI overbought and volume spike.
# Uses 1d timeframe with 1h VWAP for precision, filtered by weekly trend and volume confirmation to reduce false signals.
# Designed to work in both bull (trend continuation) and bear (mean reversion) markets by adapting to the weekly VWAP deviation.

#!/usr/bin/env python3
name = "1d_WeeklyVWAP_MeanReversion_VolumeRSI"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ===== Weekly VWAP Deviation (HTF) =====
    df_1w = get_htf_data(prices, '1w')
    # Calculate typical price and VWAP for weekly data
    typical_price_1w = (df_1w['high'].values + df_1w['low'].values + df_1w['close'].values) / 3.0
    vwap_1w = (typical_price_1w * df_1w['volume'].values).cumsum() / df_1w['volume'].values.cumsum()
    vwap_1w = np.where(df_1w['volume'].values.cumsum() == 0, np.nan, vwap_1w)
    # Align weekly VWAP to daily timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # ===== Daily RSI (14) =====
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== Daily Volume Spike (20-period average) =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)
    
    # ===== Weekly Trend Filter (EMA 21) =====
    ema21_1w = pd.Series(df_1w['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Precompute hour filter for 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure RSI and volume avg are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_avg[i]) or
            np.isnan(ema21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate deviation from weekly VWAP as percentage
        if vwap_1w_aligned[i] <= 0:
            deviation = 0
        else:
            deviation = (close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i]
        
        if position == 0:
            # Long: Price significantly below weekly VWAP, RSI oversold, volume spike, and above weekly EMA (bullish bias)
            if (deviation < -0.03 and  # 3% below VWAP
                rsi[i] < 30 and
                vol_spike[i] and
                close[i] > ema21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price significantly above weekly VWAP, RSI overbought, volume spike, and below weekly EMA (bearish bias)
            elif (deviation > 0.03 and   # 3% above VWAP
                  rsi[i] > 70 and
                  vol_spike[i] and
                  close[i] < ema21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price returns to VWAP or RSI overbought
            if (deviation > -0.01 or  # Within 1% of VWAP
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price returns to VWAP or RSI oversold
            if (deviation < 0.01 or   # Within 1% of VWAP
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals