#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with 1d trend filter and volume confirmation
# - Williams %R(14) on 6h: oversold < -80, overbought > -20
# - Trend filter: 1d EMA50 - price > 2% = uptrend bias, < -2% = downtrend bias
# - Volume confirmation: current 6h volume > 1.8x 20-period average
# - Entry logic: 
#   * Long: Williams %R < -80 AND price > 1d EMA50 AND volume spike
#   * Short: Williams %R > -20 AND price < 1d EMA50 AND volume spike
# - Exit: Williams %R crosses back above -50 (long) or below -50 (short)
# - Weekly trend filter: avoid counter-trend trades when price > weekly EMA200 for shorts or < weekly EMA200 for longs
# - Discrete position sizing (0.25) to minimize fee churn
# - Williams %R is a momentum oscillator that works well in ranging markets with trend filter
# - Target: 20-30 trades/year (80-120 total over 4 years) to stay within HARD MAX: 300 total

name = "6h_1w_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 50 or len(df_1w) < 20:
        return np.zeros(n)
    
    # Pre-compute 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Pre-compute weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Pre-compute 6h Williams %R(14)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_6h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_6h).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                         ((highest_high - close_6h) / (highest_high - lowest_low)) * -100, 
                         -50)
    
    # Pre-compute 6h volume and its 20-period moving average
    volume_6h = prices['volume'].values
    volume_ma_20_6h = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Get current values
        wr = williams_r[i]
        price = close_6h[i]
        ema_50 = ema_50_aligned[i]
        ema_200 = ema_200_aligned[i]
        volume_current = volume_6h[i]
        volume_ma = volume_ma_20_6h[i]
        
        # Volume confirmation: current 6h volume > 1.8x 20-period average
        volume_spike = volume_current > 1.8 * volume_ma
        
        # Trend filters
        price_above_ema50 = price > ema_50
        price_below_ema50 = price < ema_50
        weekly_uptrend_bias = price > ema_200  # Avoid shorts in strong weekly uptrend
        weekly_downtrend_bias = price < ema_200  # Avoid longs in strong weekly downtrend
        
        if position == 0:  # Flat - look for new entries
            # Long conditions: oversold + price above EMA50 + volume spike + weekly bias
            if (wr < -80 and price_above_ema50 and volume_spike and weekly_uptrend_bias):
                position = 1
                signals[i] = 0.25
            # Short conditions: overbought + price below EMA50 + volume spike + weekly bias
            elif (wr > -20 and price_below_ema50 and volume_spike and weekly_downtrend_bias):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            if position == 1:  # Long position
                # Exit when Williams %R crosses back above -50 (momentum fading)
                if wr > -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1, Short position
                # Exit when Williams %R crosses back below -50 (momentum fading)
                if wr < -50:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals