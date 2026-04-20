#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend and volatility regime
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility regime
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 1d Volume ratio (current / 20-period average) for confirmation
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / np.where(vol_ma_20_1d == 0, 1, vol_ma_20_1d)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # 1d Bollinger Bands width ratio for regime detection
    bb_period = 20
    bb_std = 2.0
    bb_middle = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close_1d).rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + bb_std * bb_std_dev
    bb_lower = bb_middle - bb_std * bb_std_dev
    bb_width = bb_upper - bb_lower
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    bb_width_ratio = bb_width / np.where(bb_width_ma == 0, 1, bb_width_ma)
    bb_width_ratio_aligned = align_htf_to_ltf(prices, df_1d, bb_width_ratio)
    
    # 12h price data
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(bb_width_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        bb_width_ratio = bb_width_ratio_aligned[i]
        
        # Trend filter: price relative to daily EMA
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Regime filter: avoid extreme chop (BB width too narrow)
        regime_filter = bb_width_ratio > 0.8  # Not in extreme squeeze
        
        # Volatility filter: avoid extreme volatility spikes
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.3 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio > 1.3)
        
        if position == 0:
            # Enter long in uptrend with volume and regime filter
            if trend_up and vol_filter and regime_filter:
                signals[i] = 0.25
                position = 1
            # Enter short in downtrend with volume and regime filter
            elif trend_down and vol_filter and regime_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal, volatility spike, or regime breakdown
            if not trend_up or (atr > 3.5 * atr_ma_20) or (bb_width_ratio < 0.6):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal, volatility spike, or regime breakdown
            if not trend_down or (atr > 3.5 * atr_ma_20) or (bb_width_ratio < 0.6):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_EMA50_VolumeRegime_Filter_v1"
timeframe = "12h"
leverage = 1.0