#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d timeframe with weekly trend filter, weekly volatility regime, and daily price action.
# Uses weekly Bollinger Bands to detect regime (squeeze = range, expansion = trend) and weekly close above/below SMA20 for trend.
# Entry on daily close crossing weekly Bollinger Bands with volume confirmation.
# Designed to work in both bull and bear markets by adapting to volatility regime.
# Target: 20-50 trades over 4 years (5-12/year) to minimize fee drag.

name = "1d_1w_Bollinger_Regime_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for regime and trend filters
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly close for trend and Bollinger Bands
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_volume = df_1w['volume'].values
    
    # Weekly SMA20 for trend filter
    weekly_close_series = pd.Series(weekly_close)
    sma20_1w = weekly_close_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly Bollinger Bands (20, 2.0)
    sma20_bb = weekly_close_series.rolling(window=20, min_periods=20).mean().values
    std20_bb = weekly_close_series.rolling(window=20, min_periods=20).std().values
    upper_bb = sma20_bb + 2 * std20_bb
    lower_bb = sma20_bb - 2 * std20_bb
    
    # Weekly Bollinger Band Width for regime detection (squeeze < 0.05 = low volatility)
    bb_width = (upper_bb - lower_bb) / sma20_bb
    # Regime: 1 = trend (expansion), 0 = range (squeeze)
    regime_trend = (bb_width > 0.05).astype(float)
    
    # Align weekly indicators to daily
    sma20_1w_daily = align_htf_to_ltf(prices, df_1w, sma20_1w)
    upper_bb_daily = align_htf_to_ltf(prices, df_1w, upper_bb)
    lower_bb_daily = align_htf_to_ltf(prices, df_1w, lower_bb)
    regime_trend_daily = align_htf_to_ltf(prices, df_1w, regime_trend)
    
    # Daily volume filter: volume > 1.5 * 20-day average
    volume_series = pd.Series(volume)
    vol_ma_20d = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma_20d * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for weekly SMA20 and Bollinger Bands
    
    for i in range(start_idx, n):
        if (np.isnan(sma20_1w_daily[i]) or np.isnan(upper_bb_daily[i]) or 
            np.isnan(lower_bb_daily[i]) or np.isnan(regime_trend_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        sma20 = sma20_1w_daily[i]
        upper_bb = upper_bb_daily[i]
        lower_bb = lower_bb_daily[i]
        regime = regime_trend_daily[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Enter long: price above upper BB, in trend regime, with volume
            if close[i] > upper_bb and regime > 0.5 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: price below lower BB, in trend regime, with volume
            elif close[i] < lower_bb and regime > 0.5 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below weekly SMA20 (trend reversal)
            if close[i] < sma20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above weekly SMA20 (trend reversal)
            if close[i] > sma20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals