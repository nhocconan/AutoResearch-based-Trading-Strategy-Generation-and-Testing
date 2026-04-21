#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for regime and structure
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_20 = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    atr_ratio = atr_14 / atr_ma_20
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1d Bollinger Bands for regime detection
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    bb_width_aligned = align_htf_to_ltf(prices, df_1d, bb_width)
    
    # 4h Donchian(20) for entry signals (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donch_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(bb_width_aligned[i]) or np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        ema_trend = ema_34_1d_aligned[i]
        atr_ratio_val = atr_ratio_aligned[i]
        bb_width_val = bb_width_aligned[i]
        upper_band = donch_high[i]
        lower_band = donch_low[i]
        
        # Regime filter: only trade when volatility is moderate (not too high, not too low)
        vol_regime = (atr_ratio_val > 0.8) and (atr_ratio_val < 2.5)
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high, above daily EMA34, moderate volatility
            if (price_close > upper_band and 
                price_close > ema_trend and 
                vol_regime):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 4h Donchian low, below daily EMA34, moderate volatility
            elif (price_close < lower_band and 
                  price_close < ema_trend and 
                  vol_regime):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: reverse breakout or volatility extremes
            if position == 1 and (price_low < lower_band or atr_ratio_val > 3.0):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_high > upper_band or atr_ratio_val > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_DonchianBreakout_1dTrend_VolatilityRegime"
timeframe = "4h"
leverage = 1.0