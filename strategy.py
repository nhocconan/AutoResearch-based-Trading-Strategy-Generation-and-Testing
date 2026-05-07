#!/usr/bin/env python3
name = "1d_WilliamsAlligator_ElderRay_1wTrend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def williams_alligator(high, low, close, jaw_period=13, teeth_period=8, lips_period=5):
    """Williams Alligator: returns (jaw, teeth, lips)"""
    median_price = (high + low) / 2
    jaw = pd.Series(median_price).rolling(window=jaw_period, center=False).mean().shift(8).values
    teeth = pd.Series(median_price).rolling(window=teeth_period, center=False).mean().shift(5).values
    lips = pd.Series(median_price).rolling(window=lips_period, center=False).mean().shift(3).values
    return jaw, teeth, lips

def elder_ray(high, low, close, ema_period=13):
    """Elder Ray: returns (bull_power, bear_power)"""
    ema = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    bull_power = high - ema
    bear_power = low - ema
    return bull_power, bear_power

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    trend_up = close > ema_21_1w_aligned
    trend_down = close < ema_21_1w_aligned
    
    # Williams Alligator
    jaw, teeth, lips = williams_alligator(high, low, close)
    
    # Elder Ray (13-period)
    bull_power, bear_power = elder_ray(high, low, close, 13)
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 2  # 2 days cooldown to reduce trades
    
    start_idx = max(20, 13)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine Alligator alignment (all three lines aligned)
        alligator_long = (lips[i] > teeth[i]) and (teeth[i] > jaw[i])
        alligator_short = (lips[i] < teeth[i]) and (teeth[i] < jaw[i])
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Alligator aligned up + Bull Power > 0 + volume confirmation + 1w uptrend
            if (alligator_long and 
                bull_power[i] > 0 and 
                vol_confirm[i] and 
                trend_up[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Alligator aligned down + Bear Power < 0 + volume confirmation + 1w downtrend
            elif (alligator_short and 
                  bear_power[i] < 0 and 
                  vol_confirm[i] and 
                  trend_down[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Alligator alignment changes or Bull Power turns negative
            if not alligator_long or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Alligator alignment changes or Bear Power turns positive
            if not alligator_short or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Williams Alligator identifies trend alignment (lips/teeth/jaw order), Elder Ray measures bull/bear power via price-EMA divergence, and 1w EMA21 trend filter ensures higher timeframe alignment. Volume confirmation (1.5x 20-day average) filters for institutional participation. This combination works in both bull and bear markets by capturing strong trending moves with institutional backing. The cooldown period reduces trade frequency to target 30-100 trades over 4 years. Uses discrete position sizing (0.25) to balance risk and minimize fee churn. Williams Alligator and Elder Ray are proven technical tools that have shown effectiveness in trending markets, and combining them with higher timeframe trend and volume confirmation creates a robust strategy for BTC/ETH.