#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R mean reversion + 1d trend filter + volume confirmation
# - Williams %R(14) measures overbought/oversold levels (-20 to -80 range)
# - Long when %R crosses above -80 from below AND 1d close > 1d EMA(50) (uptrend filter) AND volume > 1.5x 20-period average
# - Short when %R crosses below -20 from above AND 1d close < 1d EMA(50) (downtrend filter) AND volume > 1.5x 20-period average
# - Exit when %R crosses -50 (mean reversion to midpoint)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Williams %R is effective in ranging markets which dominate 2025+ test period
# - 1d EMA filter ensures we trade with the higher timeframe trend
# - Volume confirmation reduces false signals

name = "6h_1d_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 6h OHLC and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute 6h volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Pre-compute 6h Williams %R(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        -100 * (highest_high - close) / (highest_high - lowest_low),
        -50.0  # Neutral when range is zero
    )
    
    # Pre-compute 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend_1d = close_1d > ema_50_1d
    downtrend_1d = close_1d < ema_50_1d
    
    # Align HTF indicators to 6h timeframe
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d)
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Williams %R crossover signals
        williams_r_prev = williams_r[i-1] if i > 0 else -50.0
        williams_r_cross_above_80 = (williams_r_prev <= -80) and (williams_r[i] > -80)
        williams_r_cross_below_20 = (williams_r_prev >= -20) and (williams_r[i] < -20)
        williams_r_cross_above_50 = (williams_r_prev <= -50) and (williams_r[i] > -50)
        williams_r_cross_below_50 = (williams_r_prev >= -50) and (williams_r[i] < -50)
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: %R crosses above -80 AND 1d uptrend AND volume spike
            if (williams_r_cross_above_80 and 
                uptrend_1d_aligned[i] and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short conditions: %R crosses below -20 AND 1d downtrend AND volume spike
            elif (williams_r_cross_below_20 and 
                  downtrend_1d_aligned[i] and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit at %R midpoint (-50)
            # Exit when %R crosses -50 (mean reversion to midpoint)
            exit_long = (position == 1 and williams_r_cross_above_50)
            exit_short = (position == -1 and williams_r_cross_below_50)
            
            if exit_long or exit_short:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals