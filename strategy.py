#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band squeeze breakout with 1d ADX trend filter and volume confirmation.
# Long when price breaks above upper BB AND 1d ADX > 25 (trending) AND volume > 2.0x 20-bar average.
# Short when price breaks below lower BB AND 1d ADX > 25 (trending) AND volume > 2.0x 20-bar average.
# Uses Bollinger Band width percentile to detect squeeze (low volatility) before breakout.
# Primary timeframe: 6h, HTF: 1d for ADX trend filter.
# Target: 50-150 total trades over 4 years (12-37/year). Discrete sizing 0.25 to manage drawdown and fee drag.

name = "6h_BB_Squeeze_Breakout_1dADX25_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_s.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_ma + (bb_std_dev * bb_std)
    bb_lower = bb_ma - (bb_std_dev * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Band Width percentile (50 lookback) to detect squeeze
    bb_width_s = pd.Series(bb_width)
    bb_width_percentile = bb_width_s.rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    
    # 1d ADX calculation (14 period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(0, high[i] - high[i-1])
            minus_dm[i] = max(0, low[i-1] - low[i])
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        
        # Handle division by zero and NaN
        adx = np.where((plus_di + minus_di) == 0, 0, adx)
        return np.nan_to_num(adx, nan=0.0)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for BB and indicators
    
    for i in range(start_idx, n):
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or \
           np.isnan(bb_width_percentile[i]) or np.isnan(adx_14_aligned[i]) or \
           np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        bb_squeeze = bb_width_percentile[i] < 20  # Low volatility squeeze (bottom 20%)
        strong_trend = adx_14_aligned[i] > 25  # Strong trend filter
        
        # Bollinger Band breakout signals
        breakout_up = curr_high > bb_upper[i]  # break above upper BB
        breakout_down = curr_low < bb_lower[i]  # break below lower BB
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above upper BB AND BB squeeze AND strong trend AND volume confirmation
            if (breakout_up and 
                bb_squeeze and 
                strong_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower BB AND BB squeeze AND strong trend AND volume confirmation
            elif (breakout_down and 
                  bb_squeeze and 
                  strong_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below middle BB OR trend weakens (ADX < 20)
            if (curr_close < bb_ma[i] or 
                adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above middle BB OR trend weakens (ADX < 20)
            if (curr_close > bb_ma[i] or 
                adx_14_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals