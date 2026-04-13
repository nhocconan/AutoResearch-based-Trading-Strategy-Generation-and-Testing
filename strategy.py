#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian channel breakout with 1w volume confirmation and trend filter
    # Long: price breaks above 20-period 1w Donchian high + volume > 1.3x 20-period 1w avg + close > 50-period 1w EMA (uptrend)
    # Short: price breaks below 20-period 1w Donchian low + volume > 1.3x 20-period 1w avg + close < 50-period 1w EMA (downtrend)
    # Exit: price returns to 20-period 1w Donchian midpoint (mean reversion within channel)
    # Uses 1d primary timeframe for lower frequency and reduced fee drag
    # Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
    # Works in bull/bear: trend filter ensures we only trade with the 1w trend, avoiding false breakouts in ranging markets
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1w data for HTF indicators
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values if 'volume' in df_1w.columns else np.zeros(len(df_1w))
    
    # Calculate 20-period Donchian channels on 1w data
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    donchian_mid_20 = (donchian_high_20 + donchian_low_20) / 2
    
    # Calculate 20-period volume average on 1w data
    vol_avg_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 50-period EMA on 1w data for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid_20)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_20_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start from 50 to have enough data for calculations
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(vol_avg_20_aligned[i]) or 
            np.isnan(ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1w volume > 1.3x 20-period 1w average
        curr_vol_1w = align_htf_to_ltf(prices, df_1w, volume_1w)[i]
        volume_confirmed = curr_vol_1w > 1.3 * vol_avg_20_aligned[i]
        
        # Trend filter: close > 50-period EMA for uptrend, close < 50-period EMA for downtrend
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        # Breakout conditions
        breakout_long = (close[i] > donchian_high_aligned[i] and 
                        volume_confirmed and 
                        is_uptrend)
        breakout_short = (close[i] < donchian_low_aligned[i] and 
                         volume_confirmed and 
                         is_downtrend)
        
        # Exit conditions: return to Donchian midpoint
        exit_long = position == 1 and close[i] <= donchian_mid_aligned[i]
        exit_short = position == -1 and close[i] >= donchian_mid_aligned[i]
        
        # Execute signals
        if breakout_long and position != 1:
            position = 1
            signals[i] = position_size
        elif breakout_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_volume_trend_filter_v1"
timeframe = "1d"
leverage = 1.0