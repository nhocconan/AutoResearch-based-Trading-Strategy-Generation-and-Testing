#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1w Williams %R extreme + 1d EMA50 trend filter + volume confirmation.
# Long when Williams %R < -80 (oversold) AND close > EMA50 (uptrend) AND volume > 1.5x average.
# Short when Williams %R > -20 (overbought) AND close < EMA50 (downtrend) AND volume > 1.5x average.
# Exit when Williams %R reverses to > -50 for longs or < -50 for shorts, or price crosses EMA50.
# Uses discrete position size 0.25. Williams %R identifies overextended moves.
# 1d EMA50 ensures trading with higher timeframe trend to avoid whipsaws.
# Volume confirmation ensures moves have participation.
# 1d timeframe targets 30-100 total trades over 4 years (7-25/year) to minimize fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data once before loop for Williams %R (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get 1d data once before loop for EMA50 and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === 1w Indicators: Williams %R (14-period) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    williams_r_1w = ((highest_high_14 - close_1w) / (highest_high_14 - lowest_low_14)) * -100
    
    # === 1d Indicators: EMA50 for trend filter ===
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 1d Indicators: Volume average (20-period) for confirmation ===
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to primary timeframe (1d)
    williams_r_aligned = align_htf_to_ltf(prices, df_1w, williams_r_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60  # EMA50 and volume average need sufficient warmup
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_avg_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values (aligned)
        williams_r = williams_r_aligned[i]
        ema50 = ema50_aligned[i]
        vol_avg = vol_avg_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = vol > (1.5 * vol_avg)
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit when Williams %R > -50 (reversal from oversold) OR price < EMA50 (trend break)
            if (williams_r > -50) or (price < ema50):
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit when Williams %R < -50 (reversal from overbought) OR price > EMA50 (trend break)
            if (williams_r < -50) or (price > ema50):
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Williams %R < -80 (oversold) AND price > EMA50 (uptrend) AND volume confirmed
            if (williams_r < -80) and (price > ema50) and volume_confirmed:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Williams %R > -20 (overbought) AND price < EMA50 (downtrend) AND volume confirmed
            elif (williams_r > -20) and (price < ema50) and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25  # maintain position
    
    return signals

name = "1d_1wWilliamsR_Extreme_1dEMA50_VolumeConfirmation_V1"
timeframe = "1d"
leverage = 1.0