#!/usr/bin/env python3
# mtf_1h_hull_rsi_divergence_v1
# Hypothesis: 1h Hull Moving Average (HMA) with 4h RSI divergence filter and volume confirmation.
# Uses 1h timeframe for entry timing with 4h trend direction from HMA crossover.
# 4h RSI divergence (price making new high/low while RSI does not) filters false breakouts.
# Volume spike confirms institutional participation. Designed for 15-37 trades/year (60-150 over 4 years).
# Works in bull/bear markets: HMA captures trends with less lag, RSI divergence avoids exhaustion moves,
# volume filter ensures conviction. Session filter (08-20 UTC) reduces off-hours noise.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hull_rsi_divergence_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(series, period):
    """Calculate Hull Moving Average"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    wma_half = pd.Series(series).ewm(span=half_period, adjust=False).mean().values
    # WMA for full period
    wma_full = pd.Series(series).ewm(span=period, adjust=False).mean().values
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    # Final HMA: WMA of raw_hma with sqrt_period
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
    return hma

def calculate_rsi(series, period):
    """Calculate Relative Strength Index"""
    if len(series) < period + 1:
        return np.full_like(series, np.nan, dtype=float)
    delta = pd.Series(series).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HMA trend and RSI divergence
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough for HMA(30) and RSI(14)
        return np.zeros(n)
    
    # Calculate 4h HMA(30) for trend
    close_4h = df_4h['close'].values
    hma_4h = calculate_hma(close_4h, 30)
    
    # Align 4h HMA to 1h timeframe (completed 4h candle only)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 4h RSI(14) for divergence detection
    rsi_4h = calculate_rsi(close_4h, 14)
    
    # Align 4h RSI to 1h timeframe
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Volume spike detection (20-period volume average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 2.0)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 4h HMA OR RSI shows bearish divergence (price higher high, RSI lower high)
            if close[i] < hma_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif (i >= 2 and 
                  high[i] > high[i-1] and high[i-1] > high[i-2] and  # Price making higher high
                  rsi_4h_aligned[i] < rsi_4h_aligned[i-1] and rsi_4h_aligned[i-1] < rsi_4h_aligned[i-2]):  # RSI making lower high
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h HMA OR RSI shows bullish divergence (price lower low, RSI higher low)
            if close[i] > hma_4h_aligned[i]:
                position = 0
                signals[i] = 0.0
            elif (i >= 2 and 
                  low[i] < low[i-1] and low[i-1] < low[i-2] and  # Price making lower low
                  rsi_4h_aligned[i] > rsi_4h_aligned[i-1] and rsi_4h_aligned[i-1] > rsi_4h_aligned[i-2]):  # RSI making higher low
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price closes above 4h HMA, bullish RSI divergence, with volume spike
            if (i >= 2 and
                close[i] > hma_4h_aligned[i] and
                low[i] < low[i-1] and low[i-1] < low[i-2] and  # Price making lower low
                rsi_4h_aligned[i] > rsi_4h_aligned[i-1] and rsi_4h_aligned[i-1] > rsi_4h_aligned[i-2] and  # RSI making higher low (bullish div)
                vol_spike[i]):
                position = 1
                signals[i] = 0.20
            # Enter short: price closes below 4h HMA, bearish RSI divergence, with volume spike
            elif (i >= 2 and
                  close[i] < hma_4h_aligned[i] and
                  high[i] > high[i-1] and high[i-1] > high[i-2] and  # Price making higher high
                  rsi_4h_aligned[i] < rsi_4h_aligned[i-1] and rsi_4h_aligned[i-1] < rsi_4h_aligned[i-2] and  # RSI making lower high (bearish div)
                  vol_spike[i]):
                position = -1
                signals[i] = -0.20
    
    return signals