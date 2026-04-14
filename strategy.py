#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with 1-day trend filter and volume confirmation
# Long when Alligator jaws are above teeth and lips (bullish alignment) with price above jaws,
# volume above average, and daily EMA bullish. Short when jaws below teeth and lips (bearish
# alignment) with price below jaws, volume above average, and daily EMA bearish.
# Exit when price crosses the Alligator teeth line (middle line).
# Uses Williams Alligator (SMAs with specific periods) to identify trend phases and avoid
# choppy markets. Williams Alligator is effective in both bull and bear markets as it
# identifies when the market is "sleeping" (no trade) vs "awake" (trend).
# Target: 50-150 trades over 4 years (~12-38/year) to balance signal quality and fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h and daily data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_daily = get_htf_data(prices, '1d')
    
    # Calculate Williams Alligator on 4h timeframe
    # Jaw (Blue Line): 13-period SMMA, shifted 8 bars forward
    # Teeth (Red Line): 8-period SMMA, shifted 5 bars forward
    # Lips (Green Line): 5-period SMMA, shifted 3 bars forward
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    median_price_4h = (high_4h + low_4h) / 2  # Using median price as per Williams Alligator
    
    def smma(series, period):
        """Smoothed Moving Average (SMMA) - same as RMA/Wilder's MA"""
        if len(series) < period:
            return np.full_like(series, np.nan, dtype=float)
        result = np.full_like(series, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(series[:period])
        # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Value) / period
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    jaw = smma(median_price_4h, 13)
    teeth = smma(median_price_4h, 8)
    lips = smma(median_price_4h, 5)
    
    # Shift the lines as per Williams Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # After rolling, set the first values to NaN as they represent invalid data
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Calculate daily EMA for trend filter
    close_daily = df_daily['close'].values
    ema_daily = pd.Series(close_daily).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate 4h volume average
    vol_4h = df_4h['volume'].values
    vol_ma_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 4h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_4h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_4h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_4h, lips)
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    vol_ma_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_4h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (need enough for SMMA and shifts)
    start = 50  # conservative start to ensure all indicators are valid
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_daily_aligned[i]) or 
            np.isnan(vol_ma_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_4h_current = volume[i]
        
        if position == 0:
            # Bullish alignment: Jaw > Teeth > Lips (alligator waking up and opening mouth to eat)
            bullish_alignment = (jaw_aligned[i] > teeth_aligned[i] and 
                                teeth_aligned[i] > lips_aligned[i])
            # Bearish alignment: Jaw < Teeth < Lips (alligator waking up and opening mouth to bite)
            bearish_alignment = (jaw_aligned[i] < teeth_aligned[i] and 
                                teeth_aligned[i] < lips_aligned[i])
            
            # Long setup: bullish alignment, price above jaws, volume spike, daily bullish trend
            if (bullish_alignment and 
                price > jaw_aligned[i] and 
                vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume spike
                price > ema_daily_aligned[i]):                    # Price above daily EMA for bullish trend
                position = 1
                signals[i] = position_size
            # Short setup: bearish alignment, price below jaws, volume spike, daily bearish trend
            elif (bearish_alignment and 
                  price < jaw_aligned[i] and 
                  vol_4h_current > 1.5 * vol_ma_4h_aligned[i] and  # Volume spike
                  price < ema_daily_aligned[i]):                    # Price below daily EMA for bearish trend
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below teeth line (or bearish alignment forms)
            if (price < teeth_aligned[i] or 
                (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above teeth line (or bullish alignment forms)
            if (price > teeth_aligned[i] or 
                (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsAlligator_DailyTrend_Volume"
timeframe = "4h"
leverage = 1.0