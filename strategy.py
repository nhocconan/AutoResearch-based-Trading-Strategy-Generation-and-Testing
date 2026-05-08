#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA trend filter and volume confirmation
# Uses Williams Alligator (Jaw/Teeth/Lips) on 12h to identify trend direction,
# confirmed by 1d EMA34 trend and 1d volume > 1.5x 20-day EMA
# Avoids whipsaws in choppy markets by requiring all three lines to be aligned
# Designed for 12h timeframe to target 15-35 trades/year (60-140 total over 4 years)
# Williams Alligator is effective in both bull and bear markets as it identifies
# the absence of trend (alligator sleeping) vs presence (alligator awakening)

name = "12h_Williams_Alligator_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for trend and volume filters
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume EMA (20-period) for volume filter
    vol_ema_20 = pd.Series(df_daily['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily indicators to 12h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_daily, ema_34)
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_daily, vol_ema_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Pre-compute 12h Williams Alligator (13,8,5)
    # Jaw: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead  
    # Lips: 5-period SMMA, 3 bars ahead
    def smma(data, period):
        """Smoothed Moving Average (SmMA)"""
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Calculate SMMA for different periods
    smma_13 = smma(close, 13)
    smma_8 = smma(close, 8)
    smma_5 = smma(close, 5)
    
    # Shift to get Alligator lines (Jaw/Teeth/Lips)
    jaw = np.roll(smma_13, 8)   # 13-period SMMA shifted 8 bars ahead
    teeth = np.roll(smma_8, 5)  # 8-period SMMA shifted 5 bars ahead
    lips = np.roll(smma_5, 3)   # 5-period SMMA shifted 3 bars ahead
    
    # Pre-compute 12h ATR for stoploss (2-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=2, adjust=False, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 13)  # warmup for EMA34 and Alligator
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(ema_34_aligned[i]) or np.isnan(vol_ema_20_aligned[i]) or \
           np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 1.5x 20-day EMA
        # Find the most recent completed daily bar
        idx_daily = 0
        while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
            idx_daily += 1
        idx_daily -= 1  # last completed daily bar
        
        if idx_daily < 0:
            vol_filter = False
        else:
            vol_daily_current = df_daily.iloc[idx_daily]['volume']
            vol_filter = vol_daily_current > 1.5 * vol_ema_20_aligned[i]
        
        # Williams Alligator signals:
        # Alligator sleeping (no trend): jaws, teeth, lips intertwined
        # Alligator awake (trend): lines separated in specific order
        # Bullish: Lips > Teeth > Jaw (green alignment)
        # Bearish: Lips < Teeth < Jaw (red alignment)
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Look for entry: Alligator aligned + daily trend + volume
            if alligator_bullish and close[i] > ema_34_aligned[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            elif alligator_bearish and close[i] < ema_34_aligned[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator sleeping or price crosses below teeth
            if not alligator_bullish or close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator sleeping or price crosses above teeth
            if not alligator_bearish or close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals