#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 12h EMA50 trend filter and volume spike confirmation
# Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND close > 12h EMA50 AND volume > 2.0x 20-bar avg
# Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND close < 12h EMA50 AND volume > 2.0x 20-bar avg
# Exit when Bull/Bear Power crosses zero (momentum reversal)
# Uses discrete position sizing (0.25) to minimize fee drag. Target: 15-35 trades/year on 6h.
# Works in bull markets via trend+momentum, works in bear via volume spike requirement
# which captures panic climaxes that often precede reversals. 6h timeframe balances responsiveness with low frequency.
# Elder Ray measures price relative to EMA13 (Bull Power = High - EMA13, Bear Power = Low - EMA13) to show who controls the market.

name = "6h_ElderRay_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need sufficient data for EMA50
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Elder Ray components on 6h data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: >2.0x 20-bar average volume (strict filter for appropriate trade frequency)
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13, 20)  # Need sufficient history for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        ema_trend = ema_50_12h_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        curr_close = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND close > 12h EMA50 AND volume confirmation
            if bull > 0 and bear < 0 and curr_close > ema_trend and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power < 0 AND Bull Power < 0 (bearish momentum) AND close < 12h EMA50 AND volume confirmation
            elif bear < 0 and bull < 0 and curr_close < ema_trend and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit when Bull Power crosses below zero (momentum weakening)
            if bull <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit when Bear Power crosses above zero (momentum weakening)
            if bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals