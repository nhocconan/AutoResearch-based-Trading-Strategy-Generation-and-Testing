#!/usr/bin/env python3
# Hypothesis: 1h mean reversion on Bollinger Bands with 4h trend filter (EMA50) and 1d volume regime filter. Long when price touches lower BB in 4h uptrend + high volume regime; short when price touches upper BB in 4h downtrend + high volume regime. Uses 1h for precise entry timing, 4h for trend direction, 1d for volume regime (high/low volatility). Designed to work in both bull/bear markets by fading overextended moves during high-volume regimes. Targets 15-37 trades/year on 1h timeframe.

name = "1h_BBMeanReversion_4hEMA50_1dVolRegime_v1"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1d ATR for volume regime filter (high volatility = ATR > 20-period MA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # first bar has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_1d = pd.Series(atr_1d).rolling(window=20, min_periods=20).mean().values
    high_vol_regime = atr_1d > atr_ma_1d  # True when volatility is high
    high_vol_regime_aligned = align_htf_to_ltf(prices, df_1d, high_vol_regime.astype(float))
    
    # Calculate 1h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (std * bb_std)
    lower_band = sma - (std * bb_std)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(bb_period, n):  # start after BB lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(high_vol_regime_aligned[i]) or 
            np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches lower BB, 4h uptrend (price > EMA50), high volume regime
            if (close[i] <= lower_band[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                high_vol_regime_aligned[i] > 0.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price touches upper BB, 4h downtrend (price < EMA50), high volume regime
            elif (close[i] >= upper_band[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  high_vol_regime_aligned[i] > 0.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses above SMA (mean reversion complete) OR 4h trend breaks
            if (close[i] >= sma[i] or 
                close[i] <= ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price crosses below SMA (mean reversion complete) OR 4h trend breaks
            if (close[i] <= sma[i] or 
                close[i] >= ema_50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals