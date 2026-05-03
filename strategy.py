#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with 1w EMA50 trend filter and volume confirmation.
# Long when Bull Power > 0 (close > 13-period EMA) in weekly uptrend (close > 1w EMA50) with volume > 1.5x 20-period MA.
# Short when Bear Power < 0 (close < 13-period EMA) in weekly downtrend (close < 1w EMA50) with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Weekly EMA50 provides robust trend filter to avoid counter-trend trades across market cycles.
# Volume confirmation ensures moves have substance, reducing false signals in choppy markets.
# Target: 50-150 total trades over 4 years (12-37/year) with Sharpe > 0 on BTC/ETH/SOL.

name = "6h_ElderRay_1wEMA50_Volume"
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 13-period EMA for Elder Ray (6h data)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # Bull Power: high - EMA13
    bear_power = low - ema_13   # Bear Power: low - EMA13
    
    # Volume regime: current 6h volume > 1.5x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        close_val = close[i]
        ema_trend = ema_50_1w_aligned[i]
        bp = bull_power[i]
        br = bear_power[i]
        vol_spike = volume_spike[i]
        
        # Determine weekly trend regime
        is_weekly_uptrend = close_val > ema_trend
        is_weekly_downtrend = close_val < ema_trend
        
        # Entry logic
        if position == 0:
            if is_weekly_uptrend and bp > 0 and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_weekly_downtrend and br < 0 and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power >= 0 OR weekly trend reversal to downtrend
            if br >= 0 or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power <= 0 OR weekly trend reversal to uptrend
            if bp <= 0 or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals