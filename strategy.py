#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter, volume confirmation (>1.8x 24-bar avg volume), and ADX regime filter (ADX > 25 = trending). 
# Uses discrete sizing 0.25 to target 12-37 trades/year on 12h timeframe. 
# Camarilla pivot breaks capture institutional order flow; 1d EMA50 ensures higher timeframe trend alignment; 
# Volume confirmation filters low-participation breakouts; ADX filter avoids whipsaws in ranging markets.
# Designed for fewer, higher-quality trades to minimize fee drag while working in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA50_Volume_ADX_Filter_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ADX (14-period) for regime filter
    lookback_adx = 14
    plus_dm = pd.Series(high).diff()
    minus_dm = pd.Series(low).diff().multiply(-1)
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = pd.Series(high) - pd.Series(low)
    tr2 = abs(pd.Series(high) - pd.Series(close).shift(1))
    tr3 = abs(pd.Series(low) - pd.Series(close).shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.rolling(window=lookback_adx, min_periods=lookback_adx).mean()
    plus_di = 100 * (plus_dm.rolling(window=lookback_adx, min_periods=lookback_adx).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=lookback_adx, min_periods=lookback_adx).mean() / atr)
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di))) * 100
    adx = dx.rolling(window=lookback_adx, min_periods=lookback_adx).mean()
    adx_values = adx.values
    
    # Calculate Camarilla levels (R3, S3) from prior 12h candle only
    lookback_cam = 1
    prior_close = pd.Series(close).shift(lookback_cam).values
    prior_high = pd.Series(high).shift(lookback_cam).values
    prior_low = pd.Series(low).shift(lookback_cam).values
    prior_range = prior_high - prior_low
    
    camarilla_r3 = prior_close + (prior_range * 1.1 / 4)
    camarilla_s3 = prior_close - (prior_range * 1.1 / 4)
    
    # Calculate average volume for confirmation (24-period = 2 prior 12h bars)
    lookback_vol = 24
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback_adx, lookback_vol, 1)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(prior_close[i]) or np.isnan(prior_range[i]) or
            np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(adx_values[i])):
            signals[i] = 0.0
            continue
        
        # ADX regime filter: only trade in trending markets (ADX > 25)
        if adx_values[i] <= 25:
            # In ranging market, force flat
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA50, volume spike
            if (high[i] > camarilla_r3[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA50, volume spike
            elif (low[i] < camarilla_s3[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals