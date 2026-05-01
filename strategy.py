#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d volume spike and 12h EMA trend filter.
# Long when BB width < 20th percentile (squeeze) and price breaks above upper band with 1d volume > 2x 20-bar average and 12h EMA50 > EMA200.
# Short when BB width < 20th percentile (squeeze) and price breaks below lower band with 1d volume > 2x 20-bar average and 12h EMA50 < EMA200.
# Uses discrete sizing 0.25 to minimize fee churn. Designed for 6h timeframe to capture low-volatility breakouts in both bull and bear markets.
# Works in bull (buy breakouts above upper band) and bear (sell breakouts below lower band) via 12h EMA trend filter.

name = "6h_BB_Squeeze_Breakout_1dVolume_12hEMATrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 1d data ONCE before loop for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 1d volume spike: volume > 2x 20-bar simple moving average
    vol_ma_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean()
    vol_spike_1d = df_1d['volume'].values > (vol_ma_1d.values * 2.0)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 12h EMA trend: EMA50 and EMA200
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    ema_200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_200_12h)
    
    # 6h Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    ma_20 = close_s.rolling(window=20, min_periods=20).mean()
    std_20 = close_s.rolling(window=20, min_periods=20).std()
    upper_bb = ma_20 + (2 * std_20)
    lower_bb = ma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / ma_20  # Normalized width
    
    # Bollinger Band squeeze: width < 20th percentile of lookback
    bb_width_percentile = bb_width.rolling(window=50, min_periods=50).quantile(0.20)
    squeeze = bb_width < bb_width_percentile
    
    # Breakout conditions
    breakout_up = close > upper_bb
    breakout_down = close < lower_bb
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # warmup for 12h EMA200 and BB calculations
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_200_12h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or np.isnan(squeeze.iloc[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol_spike = vol_spike_1d_aligned[i]
        curr_ema_50 = ema_50_12h_aligned[i]
        curr_ema_200 = ema_200_12h_aligned[i]
        curr_squeeze = squeeze.iloc[i]
        curr_breakout_up = breakout_up.iloc[i]
        curr_breakout_down = breakout_down.iloc[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze AND breakout up AND volume spike AND 12h EMA50 > EMA200 (uptrend)
            if (curr_squeeze and 
                curr_breakout_up and 
                curr_vol_spike and 
                curr_ema_50 > curr_ema_200):
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze AND breakout down AND volume spike AND 12h EMA50 < EMA200 (downtrend)
            elif (curr_squeeze and 
                  curr_breakout_down and 
                  curr_vol_spike and 
                  curr_ema_50 < curr_ema_200):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: breakout below middle band (mean reversion) OR loss of squeeze (volatility expansion) OR trend change
            ma_20_val = ma_20.iloc[i]
            if (curr_close < ma_20_val or 
                not curr_squeeze or 
                curr_ema_50 < curr_ema_200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: breakout above middle band (mean reversion) OR loss of squeeze (volatility expansion) OR trend change
            ma_20_val = ma_20.iloc[i]
            if (curr_close > ma_20_val or 
                not curr_squeeze or 
                curr_ema_50 > curr_ema_200):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals