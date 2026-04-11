#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout + 1d trend filter + volume confirmation
# - Camarilla levels (L3, H3) from 1d: long when price closes above H3, short when below L3
# - Trend filter: 1d EMA50 > EMA200 for long bias, EMA50 < EMA200 for short bias
# - Volume confirmation: 4h volume > 1.8x 20-period average to filter weak breakouts
# - Discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 20-50 trades/year (80-200 total over 4 years) to stay within fee drag limits
# - Camarilla pivots identify institutional support/resistance levels
# - 1d EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation filters out false breakouts
# - Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets

name = "4h_1d_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d EMA trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Pre-compute 1d trend bias (1 for uptrend, -1 for downtrend, 0 for neutral)
    trend_bias = np.zeros(len(ema_50_aligned))
    trend_bias[ema_50_aligned > ema_200_aligned] = 1
    trend_bias[ema_50_aligned < ema_200_aligned] = -1
    
    # Pre-compute 1d Camarilla levels (based on previous day's OHLC)
    # Camarilla: H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    camarilla_h3 = close_1d_arr + 1.1 * (high_1d - low_1d) / 4
    camarilla_l3 = close_1d_arr - 1.1 * (high_1d - low_1d) / 4
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Pre-compute 4h volume confirmation (20-period average)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        
        # Camarilla breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_long = price_close > camarilla_h3_aligned[i-1]  # Close above previous period's H3
        breakout_short = price_close < camarilla_l3_aligned[i-1]  # Close below previous period's L3
        
        # Trend filter from 1d
        trend_up = trend_bias[i] == 1
        trend_down = trend_bias[i] == -1
        
        # Volume confirmation: current volume > 1.8x 20-period average
        vol_confirm = volume_current > 1.8 * volume_sma_20[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: Camarilla breakout + uptrend + volume confirmation
        if breakout_long and trend_up and vol_confirm:
            enter_long = True
        
        # Short: Camarilla breakdown + downtrend + volume confirmation
        if breakout_short and trend_down and vol_confirm:
            enter_short = True
        
        # Exit conditions: opposite Camarilla breakout or trend reversal
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if Camarilla breakdown OR trend turns down
            exit_long = (price_close < camarilla_l3_aligned[i-1]) or (not trend_up)
        elif position == -1:
            # Exit short if Camarilla breakout OR trend turns up
            exit_short = (price_close > camarilla_h3_aligned[i-1]) or (not trend_down)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals