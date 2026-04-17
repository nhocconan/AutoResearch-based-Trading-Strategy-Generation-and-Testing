#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1w EMA200 trend filter + 1d RSI mean reversion.
# Uses weekly EMA200 for long-term trend, daily RSI for overbought/oversold signals,
# and 12h Choppiness Index to filter ranging vs trending markets.
# In ranging markets (CHOP > 61.8): mean reversion at RSI extremes.
# In trending markets (CHOP < 38.2): trend following with EMA200.
# Designed for low turnover (target: 12-37 trades/year) and robustness in bull/bear markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Get daily data for RSI and Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1w EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_12h = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate 14-period RSI on daily timeframe
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_12h = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Calculate 14-period Choppiness Index on 12h timeframe
    # CHOP = 100 * log10(SUM(ATR14) / (n * (MAX(HIGH14) - MIN(LOW14)))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First period has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop_raw = np.sum(atr14)  # Placeholder for correct calculation
    # Actually calculate properly:
    chop = np.zeros_like(close)
    for i in range(14, len(close)):
        sum_atr14 = np.sum(atr14[i-13:i+1]) if i >= 13 else np.sum(atr14[14:i+1])
        if np.isnan(sum_atr14) or np.isnan(max_high14[i]) or np.isnan(min_low14[i]) or (max_high14[i] - min_low14[i]) == 0:
            chop[i] = 50.0
        else:
            chop[i] = 100 * np.log10(sum_atr14 / (14 * (max_high14[i] - min_low14[i]))) / np.log10(14)
    
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # Volume filter: current volume > 1.5 * 30-period average
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(200, 30)  # Need sufficient data for EMA200 and other indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema200_12h[i]) or 
            np.isnan(rsi_12h[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(volume_ma30[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: spike > 1.5x average
        volume_filter = volume[i] > (1.5 * volume_ma30[i])
        
        # Regime filters
        chop_high = chop[i] > 61.8  # Ranging market
        chop_low = chop[i] < 38.2   # Trending market
        
        # Trend filter: price vs weekly EMA200
        price_above_ema200 = close[i] > ema200_12h[i]
        price_below_ema200 = close[i] < ema200_12h[i]
        
        if position == 0:
            # In ranging market: mean reversion at RSI extremes
            if chop_high:
                if rsi_12h[i] < 30 and price_above_ema200 and volume_filter:  # Oversold with bullish bias
                    signals[i] = 0.25
                    position = 1
                elif rsi_12h[i] > 70 and price_below_ema200 and volume_filter:  # Overbought with bearish bias
                    signals[i] = -0.25
                    position = -1
            # In trending market: follow trend with EMA200
            elif chop_low:
                if price_above_ema200 and volume_filter:  # Strong uptrend
                    signals[i] = 0.25
                    position = 1
                elif price_below_ema200 and volume_filter:  # Strong downtrend
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: RSI overbought OR chop shifts to ranging AND RSI > 50
            if (rsi_12h[i] > 70) or (chop_high and rsi_12h[i] > 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI oversold OR chop shifts to ranging AND RSI < 50
            if (rsi_12h[i] < 30) or (chop_high and rsi_12h[i] < 50):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_ChopRegime_RSI_EMA200"
timeframe = "12h"
leverage = 1.0