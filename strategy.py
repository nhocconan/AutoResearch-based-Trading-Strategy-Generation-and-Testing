#!/usr/bin/env python3
# 1d_1w_VWAP_Breakout_TrendVolume
# Hypothesis: Daily breakout above/below weekly VWAP with trend filter from weekly EMA20 and volume confirmation.
# Uses weekly VWAP as dynamic support/resistance, weekly EMA20 for trend bias, and volume surge to avoid false breakouts.
# Designed for low trade frequency (10-25/year) to minimize fee drag in both bull and bear markets.
# Weekly VWAP provides institutional reference points that work across market regimes.

name = "1d_1w_VWAP_Breakout_TrendVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for VWAP and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate weekly VWAP (typical price * volume) / cumulative volume
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap = (typical_price * df_1w['volume']).cumsum() / df_1w['volume'].cumsum()
    vwap_values = vwap.values
    
    # Weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly VWAP and EMA20 to daily timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap_values)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(vwap_aligned[i]) or
            np.isnan(ema_20_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from weekly EMA20
        bullish_trend = close[i] > ema_20_1w_aligned[i]
        bearish_trend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation (1.8x average)
        volume_surge = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above weekly VWAP in bullish trend with volume surge
            if close[i] > vwap_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: breakdown below weekly VWAP in bearish trend with volume surge
            elif close[i] < vwap_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 2.5*ATR from highest high
                if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 2.5*ATR from lowest low
                if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals