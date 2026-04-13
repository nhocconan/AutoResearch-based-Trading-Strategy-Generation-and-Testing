#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Williams %R mean reversion with 1w trend filter and volume confirmation
    # Long: Williams %R < -80 (oversold) AND 1w close > 1w EMA50 (bullish trend) AND volume > 1.5x avg
    # Short: Williams %R > -20 (overbought) AND 1w close < 1w EMA50 (bearish trend) AND volume > 1.5x avg
    # Exit: Williams %R crosses above -50 (for long) or below -50 (for short) OR volume dry-up
    # Using 1d timeframe for low trade frequency, Williams %R for mean reversion in ranging markets,
    # 1w EMA50 for trend filter to avoid counter-trend trades, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate daily Williams %R(14)
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, ((highest_high - close) / denominator) * -100, -50)
    
    # Get daily volume for confirmation (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA50
        bullish_trend = close[i] > ema_1w_aligned[i]
        bearish_trend = close[i] < ema_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Williams %R extremes + trend filter + volume confirmation
        long_entry = (williams_r[i] < -80) and bullish_trend and vol_confirm
        short_entry = (williams_r[i] > -20) and bearish_trend and vol_confirm
        
        # Exit logic: Williams %R crosses -50 midpoint OR volume dry-up
        long_exit = (williams_r[i] > -50) or not vol_confirm
        short_exit = (williams_r[i] < -50) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_williamsr_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0