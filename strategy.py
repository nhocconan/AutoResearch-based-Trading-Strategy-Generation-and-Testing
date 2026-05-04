#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band Squeeze Breakout with 1d Trend Filter and Volume Confirmation
# Bollinger Band squeeze (low volatility) precedes expansion breakouts. 1d EMA50 ensures alignment with higher timeframe trend.
# Volume confirmation filters low-conviction breakouts. Designed for 20-50 trades/year on 4h to minimize fee drag.
# Works in bull markets via long on upside breakout in uptrend and in bear markets via short on downside breakout in downtrend.

name = "4h_BollingerSqueeze_Breakout_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Bollinger Bands (20, 2.0)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean().values
    dev = close_s.rolling(window=20, min_periods=20).std().values
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    bb_width = (upper_band - lower_band) / basis  # Normalized bandwidth
    
    # Bollinger Band Squeeze: bandwidth below 20-period mean bandwidth
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma  # Low volatility condition
    
    # Volume confirmation: 20-period volume EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bollinger Band breakout above upper band AND 1d uptrend AND volume spike
            if (close[i] > upper_band[i] and  # Upside breakout
                close[i] > ema_50_aligned[i] and  # 1d uptrend
                volume_spike[i] and
                squeeze[i]):  # Only trade after squeeze (low volatility)
                signals[i] = 0.25
                position = 1
            # Short entry: Bollinger Band breakout below lower band AND 1d downtrend AND volume spike
            elif (close[i] < lower_band[i] and  # Downside breakout
                  close[i] < ema_50_aligned[i] and  # 1d downtrend
                  volume_spike[i] and
                  squeeze[i]):  # Only trade after squeeze (low volatility)
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below basis (mean reversion) OR 1d trend turns down
            if close[i] < basis[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above basis (mean reversion) OR 1d trend turns up
            if close[i] > basis[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals