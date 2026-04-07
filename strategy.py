#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Elder Ray + 12h Trend + Volume
# Hypothesis: Elder Ray (Bull/Bear Power) captures institutional buying/selling pressure.
# Combine with 12h EMA trend filter for direction alignment and volume for confirmation.
# Works in bull/bear as Elder Ray adapts to volatility and trend filter avoids counter-trend trades.
# Targets 15-25 trades/year.

name = "6h_elder_ray_12h_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    ema12_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False).mean().values
    ema12_6h = align_htf_to_ltf(prices, df_12h, ema12_12h)
    
    # 13-period EMA for Elder Ray (standard setting)
    ema13 = pd.Series(close).ewm(span=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Buying strength
    bear_power = low - ema13   # Selling strength
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema12_6h[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear power turns positive (selling pressure) OR trend turns bearish
            if bear_power[i] > 0 or close[i] < ema12_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: Bull power turns negative (buying pressure) OR trend turns bullish
            if bull_power[i] < 0 or close[i] > ema12_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Enter long: Strong buying pressure + bullish trend + volume spike
            if vol_spike[i] and bull_power[i] > 0 and close[i] > ema12_6h[i]:
                # Require bull power to be increasing (confirming strength)
                if i == 50 or bull_power[i] > bull_power[i-1]:
                    position = 1
                    signals[i] = 0.25
            # Enter short: Strong selling pressure + bearish trend + volume spike
            elif vol_spike[i] and bear_power[i] < 0 and close[i] < ema12_6h[i]:
                # Require bear power to be decreasing (confirming strength)
                if i == 50 or bear_power[i] < bear_power[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals