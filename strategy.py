# 1d_Stochastic_Confluence_Strategy
# Hypothesis: Combines Stochastic oscillator (momentum reversal) with EMA trend filter and volume confirmation
# on daily timeframe to capture mean-reversion entries in trending markets. Works in bull/bear by aligning
# with higher timeframe trend (weekly EMA) to avoid counter-trend trades. Low trade frequency (~15-25/year)
# minimizes fee drag while maintaining edge through confluence of momentum, trend, and volume.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data once for indicators
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate daily indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Stochastic Oscillator (14,3,3)
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * ((close_1d - lowest_low) / (highest_high - lowest_low))
    # Handle division by zero when high == low
    k_percent = np.where((highest_high - lowest_low) == 0, 50, k_percent)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Daily EMA(34) for trend
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Weekly EMA(20) for higher timeframe trend
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Daily ATR(14) for volatility normalization
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to lower timeframe (daily prices)
    k_percent_aligned = align_htf_to_ltf(prices, df_1d, k_percent)
    d_percent_aligned = align_htf_to_ltf(prices, df_1d, d_percent)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(k_percent_aligned[i]) or np.isnan(d_percent_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters
        daily_uptrend = close[i] > ema_34_aligned[i]
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        daily_downtrend = close[i] < ema_34_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Stochastic signals (oversold/overbought with crossover)
        stoch_oversold = (k_percent_aligned[i] < 20) and (d_percent_aligned[i] < 20)
        stoch_overbought = (k_percent_aligned[i] > 80) and (d_percent_aligned[i] > 80)
        stoch_bullish_cross = (k_percent_aligned[i] > d_percent_aligned[i]) and \
                              (k_percent_aligned[i-1] <= d_percent_aligned[i-1])
        stoch_bearish_cross = (k_percent_aligned[i] < d_percent_aligned[i]) and \
                              (k_percent_aligned[i-1] >= d_percent_aligned[i-1])
        
        # Volume confirmation
        vol_filter = volume[i] > vol_ma_aligned[i]
        
        # Entry conditions
        long_entry = stoch_oversold and stoch_bullish_cross and daily_uptrend and weekly_uptrend and vol_filter
        short_entry = stoch_overbought and stoch_bearish_cross and daily_downtrend and weekly_downtrend and vol_filter
        
        # Exit conditions: opposite stochastic cross or trend failure
        long_exit = stoch_bearish_cross or not daily_uptrend
        short_exit = stoch_bullish_cross or not daily_downtrend
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Stochastic_Confluence_Strategy"
timeframe = "1d"
leverage = 1.0