#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Ichimoku Cloud with 1d trend filter and volume confirmation
# Long when Tenkan-sen > Kijun-sen (TK Cross) AND price > Kumo (cloud) AND 1d EMA50 uptrend AND volume > 1.5x 20-bar avg
# Short when Tenkan-sen < Kijun-sen AND price < Kumo AND 1d EMA50 downtrend AND volume > 1.5x 20-bar avg
# Exit when TK Cross reverses or price re-enters Kumo
# Ichimoku provides dynamic support/resistance with trend/momentum signals proven effective on BTC/ETH
# 1d EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation filters low-conviction signals
# Target: 12-37 trades/year on 6h (50-150 total over 4 years)

name = "6h_IchimokuTK_1dEMA50_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d data
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Ichimoku components on 6h data (using prior data to avoid look-ahead)
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    period9_high = high_series.rolling(window=9, min_periods=9).max().shift(1)
    period9_low = low_series.rolling(window=9, min_periods=9).min().shift(1)
    tenkan_sen = ((period9_high + period9_low) / 2).values
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = high_series.rolling(window=26, min_periods=26).max().shift(1)
    period26_low = low_series.rolling(window=26, min_periods=26).min().shift(1)
    kijun_sen = ((period26_high + period26_low) / 2).values
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2 shifted 26 periods ahead
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2 shifted 26 periods ahead
    period52_high = high_series.rolling(window=52, min_periods=52).max().shift(1)
    period52_low = low_series.rolling(window=52, min_periods=52).min().shift(1)
    senkou_b = ((period52_high + period52_low) / 2)
    # Shift Senkou spans forward by 26 periods (cloud is plotted ahead)
    senkou_a_shifted = pd.Series(senkou_a).shift(26).values
    senkou_b_shifted = pd.Series(senkou_b).shift(26).values
    
    # Kumo (Cloud) boundaries: Senkou Span A and Senkou Span B
    upper_kumo = np.maximum(senkou_a_shifted, senkou_b_shifted)
    lower_kumo = np.minimum(senkou_a_shifted, senkou_b_shifted)
    
    # TK Cross: Tenkan-sen > Kijun-sen (bullish) or Tenkan-sen < Kijun-sen (bearish)
    tk_cross_bullish = tenkan_sen > kijun_sen
    tk_cross_bearish = tenkan_sen < kijun_sen
    
    # Price above/below cloud
    price_above_kumo = close > upper_kumo
    price_below_kumo = close < lower_kumo
    
    # 1d EMA50 trend: rising if current > previous, falling if current < previous
    ema_50_rising = ema_50_1d_aligned > np.roll(ema_50_1d_aligned, 1)
    ema_50_falling = ema_50_1d_aligned < np.roll(ema_50_1d_aligned, 1)
    # Handle first bar
    ema_50_rising[0] = False
    ema_50_falling[0] = False
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 52  # Ichimoku needs 52 periods for Senkou B
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(tenkan_sen[i]) or np.isnan(kijun_sen[i]) or 
            np.isnan(upper_kumo[i]) or np.isnan(lower_kumo[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        ema_50 = ema_50_1d_aligned[i]
        ema_50_rise = ema_50_rising[i]
        ema_50_fall = ema_50_falling[i]
        tk_bull = tk_cross_bullish[i]
        tk_bear = tk_cross_bearish[i]
        above_kumo = price_above_kumo[i]
        below_kumo = price_below_kumo[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when TK Cross bullish AND price above cloud AND 1d EMA50 rising AND volume confirmation
            if tk_bull and above_kumo and ema_50_rise and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when TK Cross bearish AND price below cloud AND 1d EMA50 falling AND volume confirmation
            elif tk_bear and below_kumo and ema_50_fall and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when TK Cross turns bearish OR price re-enters cloud
            if not tk_bull or not above_kumo:  # TK Cross bearish OR price at/below cloud top
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when TK Cross turns bullish OR price re-enters cloud
            if not tk_bear or not below_kumo:  # TK Cross bullish OR price at/above cloud bottom
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals