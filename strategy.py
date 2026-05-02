#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R Extreme with 1w EMA34 trend filter and volume confirmation
# Uses 1d primary timeframe for low trade frequency (target: 10-25/year)
# Williams %R(14) < -80 = oversold (long), > -20 = overbought (short)
# 1w EMA34 ensures alignment with weekly trend to avoid counter-trend entries
# Volume confirmation (>1.5 * 20-period EMA) filters weak breakouts
# Designed for mean reversion in ranging markets and trend continuation in strong moves
# Works in bull markets via oversold bounces and bear markets via overbought rejections
# Avoids overtrading by requiring extreme %R levels + trend alignment + volume spike

name = "1d_WilliamsR_Extreme_1wEMA34_Trend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams %R(14) on 1d
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # avoid division by zero
    
    # Volume confirmation: volume > 1.5 * 20-period EMA (1d * 20 = 20 days)
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need sufficient data for all indicators
    start_idx = max(100, 34)  # ensure 1w EMA34 and Williams %R are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1w EMA34
        bullish_bias = close[i] > ema_34_1w_aligned[i]
        bearish_bias = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias and williams_r[i] < -80 and volume_spike[i]:
                # Long: oversold in uptrend with volume confirmation
                signals[i] = 0.25
                position = 1
            elif bearish_bias and williams_r[i] > -20 and volume_spike[i]:
                # Short: overbought in downtrend with volume confirmation
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -50 (momentum fading) or price below 1w EMA34
            if williams_r[i] > -50 or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -50 (momentum fading) or price above 1w EMA34
            if williams_r[i] < -50 or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals