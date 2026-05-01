#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R crosses above -80 from below AND 1d close > EMA50 (bullish trend) AND volume > 2.0x 20-bar average.
# Short when Williams %R crosses below -20 from above AND 1d close < EMA50 (bearish trend) AND volume > 2.0x 20-bar average.
# Williams %R identifies overextended conditions for mean reversion, EMA50 filters trend alignment, volume spike confirms momentum.
# Primary timeframe: 6h, HTF: 1d for EMA trend.
# Williams %R is effective in ranging markets (common in 2025 BTC/ETH) and avoids look-ahead by using completed candle values.

name = "6h_WilliamsR_EMA50_Trend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    prev_close = df_1d['close'].values
    ema_50 = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Williams %R on 6h data: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # We calculate it on 6h data with 14-period lookback
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: current 6h volume > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA and indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)  # Volume spike threshold
        
        # Williams %R conditions
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Long: Williams %R crosses above -80 from below (oversold reversal)
        long_condition = (wr > -80) and (wr_prev <= -80)
        # Short: Williams %R crosses below -20 from above (overbought reversal)
        short_condition = (wr < -20) and (wr_prev >= -20)
        
        # Trend filter: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = curr_close > ema_50_aligned[i]
        bearish_trend = curr_close < ema_50_aligned[i]
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: oversold reversal AND bullish trend AND volume confirmation
            if (long_condition and 
                bullish_trend and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: overbought reversal AND bearish trend AND volume confirmation
            elif (short_condition and 
                  bearish_trend and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Williams %R crosses above -20 (overbought) OR trend turns bearish
            if (wr >= -20) or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Williams %R crosses below -80 (oversold) OR trend turns bullish
            if (wr <= -80) or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals