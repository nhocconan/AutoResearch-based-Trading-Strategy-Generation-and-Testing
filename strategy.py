#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Bull/Bear Power with 12h EMA50 trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Long when Bull Power > 0 and rising, price > 12h EMA50, volume > 1.5x average
# Short when Bear Power < 0 and falling, price < 12h EMA50, volume > 1.5x average
# Uses 12h EMA50 for higher timeframe trend alignment (more stable than shorter EMAs).
# Volume confirmation reduces false signals during low participation.
# Discrete position sizing at ±0.25 to minimize fee churn.
# Target: 50-120 total trades over 4 years (12-30/year) to avoid overtrading.
# Works in bull markets via sustained buying pressure and in bear markets via distribution signals.

name = "6h_ElderRay_BullBearPower_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid low liquidity
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h_vals = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h_vals).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate EMA13 for Elder Ray (using primary timeframe)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Align 12h indicators to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for EMA50 and EMA13
    
    for i in range(start_idx, n):
        # Skip if indicators not available or outside session
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i]) or
            np.isnan(volume_confirm[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_ema_50_12h = ema_50_12h_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power positive and rising, above 12h EMA50, volume confirmation
            if (curr_bull_power > 0 and 
                curr_bull_power > bull_power[i-1] and  # Rising bull power
                curr_close > curr_ema_50_12h and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative and falling, below 12h EMA50, volume confirmation
            elif (curr_bear_power < 0 and 
                  curr_bear_power < bear_power[i-1] and  # Falling bear power (more negative)
                  curr_close < curr_ema_50_12h and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit when bull power turns negative or price breaks below 12h EMA50
            if (curr_bull_power <= 0 or 
                curr_close < curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when bear power turns positive or price breaks above 12h EMA50
            if (curr_bear_power >= 0 or 
                curr_close > curr_ema_50_12h):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals