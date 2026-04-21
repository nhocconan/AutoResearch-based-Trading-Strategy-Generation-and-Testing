#!/usr/bin/env python3
"""
6h_Keltner_Channel_Breakout_12hTrend_Volume
Hypothesis: Use Keltner Channel breakout on 6h with 12h EMA trend filter and volume confirmation. 
Keltner Channels adapt to volatility, providing dynamic support/resistance. In trending markets 
(12h EMA slope), breakouts from KC with volume surge indicate strong momentum. Works in bull/bear 
by following 12h trend. Targets 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # === 12h trend filter: 34-period EMA slope ===
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_slope = pd.Series(ema_34_12h).diff(3).values  # 3-period slope for trend
    ema_slope_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h_slope)
    
    # === Keltner Channel on 6h (20 EMA, 2.0 ATR) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # EMA(20) of close
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # ATR(20)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Bands
    kc_upper = ema_20 + 2.0 * atr
    kc_lower = ema_20 - 2.0 * atr
    
    # === Volume confirmation: 20-period volume average ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup
        # Skip if indicators not ready
        if (np.isnan(ema_slope_aligned[i]) or
            np.isnan(kc_upper[i]) or
            np.isnan(kc_lower[i]) or
            np.isnan(ema_20[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        trend_slope = ema_slope_aligned[i]
        upper_band = kc_upper[i]
        lower_band = kc_lower[i]
        basis = ema_20[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above KC upper + volume spike > 1.5 + positive 12h EMA slope
            if (price_close > upper_band and 
                vol_spike > 1.5 and 
                trend_slope > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below KC lower + volume spike > 1.5 + negative 12h EMA slope
            elif (price_close < lower_band and 
                  vol_spike > 1.5 and 
                  trend_slope < 0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to KC basis (middle line)
            if position == 1 and price_close < basis:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > basis:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Keltner_Channel_Breakout_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0