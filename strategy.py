#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 12h Volume Confirmation and 1d Trend Filter
# Strategy trades volatility breakouts after low volatility periods (Bollinger Band squeeze).
# Long: Bollinger Band width < 20th percentile + price breaks above upper band + volume spike + 1d close > EMA34
# Short: Bollinger Band width < 20th percentile + price breaks below lower band + volume spike + 1d close < EMA34
# Uses 6h timeframe for lower frequency (target: 12-37 trades/year) to minimize fee drag
# Bollinger Squeeze identifies compression before expansion; volume confirms breakout; 1d EMA34 ensures higher timeframe alignment

name = "6h_BollingerSqueeze_Breakout_12hVolume_1dEMA34_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h HTF data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1d HTF data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Bollinger Bands (20, 2) on 6h
    bb_period = 20
    bb_std = 2
    sma_bb = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma_bb + (bb_std_dev * bb_std)
    lower_band = sma_bb - (bb_std_dev * bb_std)
    bb_width = (upper_band - lower_band) / sma_bb  # Normalized bandwidth
    
    # Bollinger Band squeeze: width < 20th percentile of last 50 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).quantile(0.20).values
    squeeze_condition = bb_width < bb_width_percentile
    
    # 12h volume confirmation: current 12h volume > 1.5 * 20-period average volume
    volume_12h = df_12h['volume'].values
    volume_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_ma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_20_12h)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    volume_spike_12h = volume_12h_aligned > (volume_ma_20_12h_aligned * 1.5)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(bb_period, 50, 20, 34)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(sma_bb[i]) or np.isnan(bb_std_dev[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(bb_width_percentile[i]) or 
            np.isnan(volume_ma_20_12h_aligned[i]) or np.isnan(volume_12h_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        breakout_up = close[i] > upper_band[i]
        breakout_down = close[i] < lower_band[i]
        
        # Volume and trend filters
        vol_spike = volume_spike_12h[i]
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Squeeze + breakout up + volume spike + uptrend
            if squeeze_condition[i] and breakout_up and vol_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Squeeze + breakout down + volume spike + downtrend
            elif squeeze_condition[i] and breakout_down and vol_spike and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on breakout below middle band or loss of squeeze/trend
            if close[i] < sma_bb[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on breakout above middle band or loss of squeeze/trend
            if close[i] > sma_bb[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals