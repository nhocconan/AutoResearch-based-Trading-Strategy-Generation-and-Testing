#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with daily volatility filter and volume confirmation.
# Uses 4h Donchian(20) breakout in direction of 1d EMA(50) trend.
# Requires 1d ATR > 10-period median (avoid low volatility) and volume spike (>1.5x 24-period average).
# Exit when price touches opposite Donchian band or reverses against 1d EMA.
# Designed for 20-40 trades/year with clear rules to minimize fee drag.
# Works in bull/bear by following higher timeframe trend with volatility filter.
name = "4h_Donchian20_Volume_TrendFilter_V2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for trend and volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close']
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d ATR(14) for volatility filter
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d.shift(1))
    tr3 = np.abs(low_1d - close_1d.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median_1d = pd.Series(atr_14_1d).rolling(window=10, min_periods=10).median().values
    
    # Align to 4h
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    atr_median_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_median_1d)
    
    # 4h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 24-period average (24 * 4h = 4 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (1.5 * vol_ma_24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(atr_median_1d_aligned[i]) or np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_24[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_50_1d_aligned[i]
        atr_vol = atr_14_1d_aligned[i]
        atr_med = atr_median_1d_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_filter = volume_filter[i]
        
        # Volatility filter: avoid low volatility environments
        vol_filter_ok = atr_vol > atr_med
        
        if position == 0:
            # Long: break above upper band with uptrend, volume, and volatility
            if close_val > upper_band and close_val > ema_trend and vol_filter and vol_filter_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band with downtrend, volume, and volatility
            elif close_val < lower_band and close_val < ema_trend and vol_filter and vol_filter_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches lower band or reverses against trend
            if close_val < lower_band or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches upper band or reverses against trend
            if close_val > upper_band or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals