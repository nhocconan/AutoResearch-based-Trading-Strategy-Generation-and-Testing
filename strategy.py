#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla levels from daily provide precise support/resistance; breakouts capture momentum
# 1d EMA34 ensures alignment with medium-term trend; volume >1.8x confirms participation
# Discrete sizing (0.25) minimizes fee churn; target 75-200 total trades over 4 years
# Works in bull/bear: breakouts catch momentum moves, volume filter ensures legitimacy, EMA34 trend filter avoids counter-trend trades

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility (14-period)
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr2.iloc[0] = 0
    tr3.iloc[0] = 0
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 1.8x 30-period average
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (1.8 * vol_ma_30)
    
    # Precompute daily data for Camarilla levels
    df_1d_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_camarilla) < 2:
        return np.zeros(n)
    
    daily_high = df_1d_camarilla['high'].values
    daily_low = df_1d_camarilla['low'].values
    daily_close = df_1d_camarilla['close'].values
    
    daily_high_aligned = align_htf_to_ltf(prices, df_1d_camarilla, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d_camarilla, daily_low)
    daily_close_aligned = align_htf_to_ltf(prices, df_1d_camarilla, daily_close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30, 14)  # warmup: need EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(atr[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma_30[i]) or
            np.isnan(daily_high_aligned[i]) or
            np.isnan(daily_low_aligned[i]) or
            np.isnan(daily_close_aligned[i])):
            signals[i] = 0.0
            continue
            
        curr_high = high[i]
        curr_low = low[i]
        curr_close = close[i]
        curr_volume_confirm = volume_confirm[i]
        curr_ema_34_1d = ema_34_1d_aligned[i]
        
        # Use previous day's levels (shift by 1)
        prev_high = daily_high_aligned[i-1]
        prev_low = daily_low_aligned[i-1]
        prev_close = daily_close_aligned[i-1]
        
        if position == 0:  # Flat - look for new entries
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                # Calculate Camarilla levels
                range_val = prev_high - prev_low
                r3 = prev_close + (range_val * 1.1 / 4)
                s3 = prev_close - (range_val * 1.1 / 4)
                
                # Only trade with volume confirmation and trend filter
                if curr_volume_confirm:
                    # Bullish entry: price breaks above R3 + above 1d EMA34
                    if curr_high > r3 and curr_close > curr_ema_34_1d:
                        signals[i] = 0.25
                        position = 1
                    # Bearish entry: price breaks below S3 + below 1d EMA34
                    elif curr_low < s3 and curr_close < curr_ema_34_1d:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:  # Long position
            # Exit: price breaks below S3
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                range_val = prev_high - prev_low
                s3 = prev_close - (range_val * 1.1 / 4)
                if curr_low < s3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price breaks above R3
            if not (np.isnan(prev_high) or np.isnan(prev_low) or np.isnan(prev_close)):
                range_val = prev_high - prev_low
                r3 = prev_close + (range_val * 1.1 / 4)
                if curr_high > r3:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals