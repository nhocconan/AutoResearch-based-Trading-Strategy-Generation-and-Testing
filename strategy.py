#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) identifies trend via smoothed MAs
# Long when Lips > Teeth > Jaw (bullish alignment), Short when Lips < Teeth < Jaw (bearish)
# 1d EMA34 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume > 1.5x 20-period average confirms participation
# Discrete sizing (0.25) minimizes fee churn; target 50-150 total trades over 4 years
# Works in bull/bear: Alligator catches emerging trends, volume filter ensures legitimacy,
# 1d EMA34 trend filter avoids whipsaws in ranging markets

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all smoothed with SMMA
    # SMMA is similar to EMA but with different smoothing factor
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw (blue) - 13-period SMMA
    teeth = smma(close, 8)  # Teeth (red) - 8-period SMMA
    lips = smma(close, 5)   # Lips (green) - 5-period SMMA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20, 34)  # warmup: need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(jaw[i]) or
            np.isnan(teeth[i]) or
            np.isnan(lips[i])):
            signals[i] = 0.0
            continue
            
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        curr_close = close[i]
        
        if position == 0:  # Flat - look for new entries
            if curr_volume_confirm:
                # Bullish entry: Lips > Teeth > Jaw (bullish alignment) + above 1d EMA34
                if curr_lips > curr_teeth > curr_teeth and curr_lips > curr_jaw and curr_close > curr_ema_34_1d:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: Lips < Teeth < Jaw (bearish alignment) + below 1d EMA34
                elif curr_lips < curr_teeth < curr_jaw and curr_close < curr_ema_34_1d:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position
            # Exit: Alligator lines cross (Lips < Teeth or Teeth < Jaw) - trend weakening
            if curr_lips < curr_teeth or curr_teeth < curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator lines cross (Lips > Teeth or Teeth > Jaw) - trend weakening
            if curr_lips > curr_teeth or curr_teeth > curr_jaw:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals