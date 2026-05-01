#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Squeeze Breakout with 1d EMA34 trend filter and volume confirmation
# Bollinger Squeeze occurs when BB width reaches a low percentile (low volatility), followed by breakout
# Long when price breaks above upper BB with volume spike and price > 1d EMA34
# Short when price breaks below lower BB with volume spike and price < 1d EMA34
# Uses 1d EMA34 for higher-timeframe trend alignment to reduce whipsaws in ranging markets.
# Volume spike confirms institutional participation. Discrete sizing (0.25) minimizes fee churn.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits.

name = "6h_BollingerSqueeze_Breakout_1dEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d HTF data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) on 1d close
    ema_1d_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 6h timeframe
    ema_1d_34_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_34)
    
    # Bollinger Bands (20, 2) on 6h close
    close_s = pd.Series(close)
    bb_ma = close_s.rolling(window=20, min_periods=20).mean().values
    bb_std = close_s.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_ma + (2 * bb_std)
    bb_lower = bb_ma - (2 * bb_std)
    bb_width = bb_upper - bb_lower
    
    # Bollinger Squeeze: BB width below 20th percentile of last 100 periods
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=100, min_periods=100).quantile(0.20).values
    bb_squeeze = bb_width <= bb_width_percentile
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = 100  # Need 100 for BB width percentile
    
    for i in range(start_idx, n):
        if (np.isnan(ema_1d_34_aligned[i]) or np.isnan(bb_ma[i]) or np.isnan(bb_std[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_spike = volume_spike[i]
        
        # Bollinger Breakout conditions
        breakout_up = curr_close > bb_upper[i]
        breakout_down = curr_close < bb_lower[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: BB squeeze, breakout above upper BB, price above 1d EMA34, volume spike
            if bb_squeeze[i] and breakout_up and curr_close > ema_1d_34_aligned[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze, breakout below lower BB, price below 1d EMA34, volume spike
            elif bb_squeeze[i] and breakout_down and curr_close < ema_1d_34_aligned[i] and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on break below middle BB or price below 1d EMA34
            if curr_close < bb_ma[i] or curr_close < ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on break above middle BB or price above 1d EMA34
            if curr_close > bb_ma[i] or curr_close > ema_1d_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals