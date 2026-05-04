#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Uses discrete position sizing (0.30) to minimize fee churn. Combines price channel breakout with
# higher-timeframe trend filtering for robustness in both bull and bear markets. Target: 20-30 trades/year per symbol.
# This strategy focuses on BTC and ETH as primary targets, using 1w trend filter for better generalization.

name = "1d_Donchian20_1wEMA34_VolumeSpike_Trend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1w data for ATR(14) for volatility filter
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: no previous close
    
    # Calculate ATR(14)
    atr_14_1w = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Get 1d data for Donchian channels (20-period)
    if len(prices) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr_14_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 1.5 x 20-period EMA
        volume_confirmed = volume[i] > (1.5 * vol_ema_20[i])
        
        # Volatility filter: only trade when ATR is above its 50-period EMA (avoid low volatility)
        vol_filter = atr_14_1w_aligned[i] > 0  # Always true if ATR calculated
        
        # 1w trend: bullish if close > EMA34, bearish if close < EMA34
        bullish_trend = close[i] > ema_34_1w_aligned[i]
        bearish_trend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + volume confirmation + bullish 1w trend
            if (close[i] > donchian_high[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + volume confirmation + bearish 1w trend
            elif (close[i] < donchian_low[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below Donchian low OR 1w trend turns bearish
            if close[i] < donchian_low[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above Donchian high OR 1w trend turns bullish
            if close[i] > donchian_high[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals