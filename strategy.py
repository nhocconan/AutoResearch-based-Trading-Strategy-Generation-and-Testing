#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + Elder Ray + Volume Spike with 1w trend filter.
# Long when: Alligator bullish (jaw < teeth < lips), Elder Bull Power > 0, volume > 2.0x 20-bar avg, 1w EMA34 up.
# Short when: Alligator bearish (jaw > teeth > lips), Elder Bear Power < 0, volume > 2.0x 20-bar avg, 1w EMA34 down.
# Exit when Alligator reverses (jaw > lips for long, jaw < lips for short) or Elder Power crosses zero.
# Uses discrete sizing (0.25) to limit fee drag. Target: 80-120 total trades over 4 years (20-30/year).
# Williams Alligator identifies trend via smoothed medians; Elder Ray measures bull/bear power; Volume confirms conviction.
# 1w EMA34 filters for higher-timeframe trend alignment to avoid counter-trend whipsaws.

name = "12h_WilliamsAlligator_ElderRay_VolumeSpike_1wEMA34_v1"
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
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams Alligator: SMMA (smoothed moving average) of median price
    median_price = (high + low) / 2
    
    # Jaw: SMMA(median, 13) shifted 8 bars
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8).values
    # Teeth: SMMA(median, 8) shifted 5 bars
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5).values
    # Lips: SMMA(median, 5) shifted 3 bars
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: >2.0x 20-bar average
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 1w EMA34 trend: slope over 3 periods
        if i >= 3:
            ema_slope = (ema_34_1w_aligned[i] - ema_34_1w_aligned[i-3]) / 3
            ema_trend_up = ema_slope > 0
            ema_trend_down = ema_slope < 0
        else:
            ema_trend_up = False
            ema_trend_down = False
        
        # Williams Alligator conditions
        alligator_bullish = jaw[i] < teeth[i] and teeth[i] < lips[i]
        alligator_bearish = jaw[i] > teeth[i] and teeth[i] > lips[i]
        
        # Elder Ray conditions
        elder_bull = bull_power[i] > 0
        elder_bear = bear_power[i] < 0
        
        # Entry conditions
        enter_long = alligator_bullish and elder_bull and vol_confirm and ema_trend_up
        enter_short = alligator_bearish and elder_bear and vol_confirm and ema_trend_down
        
        # Exit conditions: Alligator reverses or Elder Power crosses zero
        exit_long = not alligator_bullish or bull_power[i] <= 0
        exit_short = not alligator_bearish or bear_power[i] >= 0
        
        # Handle entries and exits
        if enter_long and position <= 0:
            signals[i] = 0.25
            position = 1
        elif enter_short and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals