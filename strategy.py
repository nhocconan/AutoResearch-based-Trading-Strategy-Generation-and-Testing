#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume confirmation.
# Bull Power = High - EMA13, Bear Power = EMA13 - Low (using 13-period EMA on 6h).
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) in uptrend (close > 12h EMA50) with volume > 1.5x 20-period MA.
# Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) in downtrend (close < 12h EMA50) with volume spike.
# Uses 6h primary timeframe with 12h HTF for trend filter. Discrete sizing 0.25.
# Elder Ray measures market power; 12h EMA50 filters counter-trend whipsaw.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "6h_ElderRay_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h EMA13 for Elder Ray (using close prices)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_12h_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_spike = volume_spike[i]
        
        # Determine trend regime
        is_uptrend = close_val > ema_trend
        is_downtrend = close_val < ema_trend
        
        # Elder Ray conditions for momentum
        is_bullish_momentum = bp > 0 and br < 0  # Bull Power positive, Bear Power negative
        is_bearish_momentum = br > 0 and bp < 0   # Bear Power positive, Bull Power negative
        
        # Entry logic
        if position == 0:
            if is_uptrend and is_bullish_momentum and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_downtrend and is_bearish_momentum and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: momentum turns bearish OR trend reversal
            if not is_bullish_momentum or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: momentum turns bullish OR trend reversal
            if not is_bearish_momentum or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals