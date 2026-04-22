#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index regime filter with 1-week EMA trend and volume confirmation
# Choppiness Index (CHOP) > 61.8 indicates ranging market (mean reversion opportunity)
# CHOP < 38.2 indicates trending market (trend following)
# In ranging markets: buy near lower Bollinger Band, sell near upper Bollinger Band
# In trending markets: follow 1-week EMA direction
# Volume confirmation (>1.3x 20-period average) filters false signals
# Designed for 1d timeframe targeting 15-25 trades/year to minimize fee drag in choppy 2025 market

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1-week EMA(50) for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Choppiness Index (14) - measures ranging vs trending markets
    # CHOP = 100 * log10(sum(ATR(14) over 14 periods) / (log10(highest high - lowest low over 14 periods)))
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with original index
    
    atr14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = highest_high14 - lowest_low14
    
    # Avoid division by zero
    range14 = np.where(range14 == 0, 1e-10, range14)
    
    chop = 100 * np.log10(sum_atr14 / range14) / np.log10(14)
    
    # Bollinger Bands (20, 2) for mean reversion in ranging markets
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_band = sma20 + 2 * std20
    lower_band = sma20 - 2 * std20
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(sma20[i]) or np.isnan(std20[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Ranging market (CHOP > 61.8): mean reversion at Bollinger Bands
            if chop[i] > 61.8:
                # Long near lower Bollinger Band
                if close[i] <= lower_band[i] and volume[i] > 1.3 * vol_avg_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Short near upper Bollinger Band
                elif close[i] >= upper_band[i] and volume[i] > 1.3 * vol_avg_20[i]:
                    signals[i] = -0.25
                    position = -1
            # Trending market (CHOP < 38.2): follow 1-week EMA trend
            elif chop[i] < 38.2:
                # Long in uptrend
                if close[i] > ema_50_1w_aligned[i] and volume[i] > 1.3 * vol_avg_20[i]:
                    signals[i] = 0.25
                    position = 1
                # Short in downtrend
                elif close[i] < ema_50_1w_aligned[i] and volume[i] > 1.3 * vol_avg_20[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: opposite Bollinger Band or trend reversal
                if (chop[i] > 61.8 and close[i] >= upper_band[i]) or \
                   (chop[i] < 38.2 and close[i] < ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: opposite Bollinger Band or trend reversal
                if (chop[i] > 61.8 and close[i] <= lower_band[i]) or \
                   (chop[i] < 38.2 and close[i] > ema_50_1w_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_ChopRegime_EMA50_Volume"
timeframe = "1d"
leverage = 1.0