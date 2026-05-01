#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA50 trend filter and volume spike confirmation.
# Long when price breaks above 4h Camarilla R1 level AND price > 12h EMA50 AND volume > 2x 20-bar average.
# Short when price breaks below 4h Camarilla S1 level AND price < 12h EMA50 AND volume > 2x 20-bar average.
# Exit on opposite Camarilla breakout (S1/R1) or when price crosses 4h EMA20.
# Uses Camarilla pivots for institutional structure (proven edge on ETH), 12h EMA for trend alignment, volume spike for conviction.
# Target: 20-50 trades/year on 4h. Discrete sizing 0.25 to minimize fee drag while capturing strong trends.
# Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend) by trading with aligned 12h trend.

name = "4h_Camarilla_R1S1_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # Load 4h data ONCE before loop for Camarilla and EMA calculations
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h typical price for Camarilla (using previous bar's OHLC)
    typical_4h = (df_4h['high'].values + df_4h['low'].values + df_4h['close'].values) / 3
    # Camarilla levels: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    # Using previous bar to avoid look-ahead
    prev_high = np.roll(df_4h['high'].values, 1)
    prev_low = np.roll(df_4h['low'].values, 1)
    prev_close = np.roll(df_4h['close'].values, 1)
    prev_high[0] = prev_low[0] = prev_close[0] = np.nan  # first bar has no previous
    
    camarilla_r1 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 12
    camarilla_s1 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h primary timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 4h EMA20 for dynamic exit
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h primary timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # warmup for EMA50 (50) + Camarilla (1) + EMA20 (20)
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC (reduce noise, focus on active sessions)
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            # Outside session: flatten position if any
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_r1 = camarilla_r1_aligned[i]
        curr_s1 = camarilla_s1_aligned[i]
        curr_ema_20 = ema_20_4h_aligned[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        
        # Volume confirmation: current 4h volume > 2x 20-period average
        vol_4h = df_4h['volume'].values
        vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
        vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
        curr_vol_ma = vol_ma_4h_aligned[i]
        volume_confirm = curr_vol > (curr_vol_ma * 2.0)
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: break above Camarilla R1 + price > 12h EMA50 + volume confirmation
            if (curr_high > curr_r1 and 
                curr_close > curr_ema_50_12h and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: break below Camarilla S1 + price < 12h EMA50 + volume confirmation
            elif (curr_low < curr_s1 and 
                  curr_close < curr_ema_50_12h and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below 20-period EMA OR breaks below Camarilla S1 (opposite)
            if (curr_close < curr_ema_20) or \
               (curr_low < curr_s1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above 20-period EMA OR breaks above Camarilla R1 (opposite)
            if (curr_close > curr_ema_20) or \
               (curr_high > curr_r1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals