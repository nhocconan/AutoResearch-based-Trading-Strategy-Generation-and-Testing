#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + Elder Ray + Volume Spike
# - Primary: 4h Williams Alligator (jaw=13, teeth=8, lips=5) for trend direction
# - HTF: 1d Elder Ray Power (bull/bear power) for institutional trend confirmation
# - Regime filter: 4h volume > 1.5x 20-period MA for participation confirmation
# - Long: Alligator bullish (lips > teeth > jaw) + Elder Ray bull power > 0 + volume confirmation
# - Short: Alligator bearish (lips < teeth < jaw) + Elder Ray bear power < 0 + volume confirmation
# - Exit: Alligator cross (lips crosses teeth) or Elder Ray power reverses sign
# - Position sizing: 0.25 (discrete level to minimize fee churn)
# - Works in bull/bear: Alligator catches trends, Elder Ray filters weak moves, volume confirms participation
# - Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe

name = "4h_1d_alligator_elder_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute primary timeframe data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute HTF data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h Williams Alligator
    # Jaw: 13-period SMMA (smoothed) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            if not np.isnan(source[i]) and not np.isnan(result[i-1]):
                result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Calculate 1d Elder Ray Power
    # Bull Power = High - EMA(close, 13)
    # Bear Power = Low - EMA(close, 13)
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = low_1d - ema_13_1d
    
    # Calculate 4h volume moving average (20-period)
    volume_ma_20 = np.full(len(volume), np.nan)
    for i in range(19, len(volume)):
        if not np.isnan(volume[i-19:i+1]).any():
            volume_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Align HTF indicators to 4h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if any required data is invalid
        if (np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period MA
        volume_confirm = volume[i] > 1.5 * volume_ma_20[i]
        
        # Alligator conditions
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Elder Ray conditions
        elder_bull = bull_power_1d_aligned[i] > 0
        elder_bear = bear_power_1d_aligned[i] < 0
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Alligator bullish + Elder Ray bull + volume confirmation
            if alligator_bullish and elder_bull and volume_confirm:
                position = 1
                signals[i] = 0.25
            # Short entry: Alligator bearish + Elder Ray bear + volume confirmation
            elif alligator_bearish and elder_bear and volume_confirm:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator cross or Elder Ray power reverses
            exit_long = not alligator_bullish or not elder_bull
            exit_short = not alligator_bearish or not elder_bear
            
            if position == 1:  # Long position
                if exit_long:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if exit_short:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals