#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 12h EMA50 trend filter and volume spike confirmation.
# Long when Bull Power > 0 (close > EMA13) AND Bear Power < 0 (low < EMA13) AND close > 12h EMA50 with volume > 1.5x 20-period MA.
# Short when Bull Power < 0 AND Bear Power > 0 AND close < 12h EMA50 with volume spike.
# Uses discrete position sizing (0.25) to minimize fee churn while maintaining sufficient exposure.
# 12h EMA50 provides higher timeframe trend filter to avoid counter-trend trades in both bull and bear markets.
# Volume confirmation ensures moves have institutional participation, reducing false signals.
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
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray components on 6h data (13-period EMA for reference)
    ema_13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema_13  # Bull Power: High - EMA13
    bear_power = low - ema_13   # Bear Power: Low - EMA13
    
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
        is_bull_trend = close_val > ema_trend
        is_bear_trend = close_val < ema_trend
        
        # Elder Ray conditions
        bull_power_pos = bp > 0   # Bull Power positive
        bear_power_neg = br < 0   # Bear Power negative
        bull_power_neg = bp < 0   # Bull Power negative
        bear_power_pos = br > 0   # Bear Power positive
        
        # Entry logic
        if position == 0:
            if is_bull_trend and bull_power_pos and bear_power_neg and vol_spike:
                signals[i] = 0.25
                position = 1
            elif is_bear_trend and bull_power_neg and bear_power_pos and vol_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Elder Ray divergence OR trend reversal
            if (bp <= 0 or br >= 0) or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Elder Ray divergence OR trend reversal
            if (bp >= 0 or br <= 0) or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals