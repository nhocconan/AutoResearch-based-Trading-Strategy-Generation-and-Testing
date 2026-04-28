#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray with 12h trend filter
# Williams Alligator (JAW/TEETH/LIPS) identifies trendless markets when lines intertwine.
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures bull/bear strength.
# Enter long when Alligator is aligned (JAW < TEETH < LIPS) and Bull Power > 0 with volume confirmation.
# Enter short when Alligator aligned inversely (JAW > TEETH > LIPS) and Bear Power > 0 with volume confirmation.
# 12h EMA(34) ensures alignment with medium-term trend to avoid counter-trend whipsaws.
# Designed for 6h timeframe targeting 12-37 trades/year to minimize fee drag while capturing strong trends.
# Works in bull markets via long signals and bear markets via short signals.

name = "6h_WilliamsAlligator_ElderRay_12hEMA34_Trend_Volume_v1"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA(34) for trend
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align 12h EMA to 6h (changes only when 12h bar closes)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Williams Alligator: SMA of median price (HL/2) with different periods
    # JAW: SMA(13, 8) - 13-period SMA shifted 8 bars ahead
    # TEETH: SMA(8, 5) - 8-period SMA shifted 5 bars ahead  
    # LIPS: SMA(5, 3) - 5-period SMA shifted 3 bars ahead
    median_price = (high + low) / 2
    
    # Calculate SMAs with shifts (using min_periods to avoid look-ahead)
    jaw_raw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().shift(8)
    teeth_raw = pd.Series(median_price).rolling(window=8, min_periods=8).mean().shift(5)
    lips_raw = pd.Series(median_price).rolling(window=5, min_periods=5).mean().shift(3)
    
    jaw = jaw_raw.values
    teeth = teeth_raw.values
    lips = lips_raw.values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume confirmation: >1.8x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.8 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34, 13, 8, 5, 14)  # volume MA, EMA34, Alligator components, ATR
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        
        # Check Alligator alignment
        # Bullish alignment: JAW < TEETH < LIPS (lines separated and rising)
        # Bearish alignment: JAW > TEETH > LIPS (lines separated and falling)
        bullish_aligned = jaw[i] < teeth[i] < lips[i]
        bearish_aligned = jaw[i] > teeth[i] > lips[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Bullish Alligator alignment, Bull Power > 0, above 12h EMA34, volume spike
            if bullish_aligned and bull_power[i] > 0 and price > ema_34_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: Bearish Alligator alignment, Bear Power > 0, below 12h EMA34, volume spike
            elif bearish_aligned and bear_power[i] > 0 and price < ema_34_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or Alligator reversal
            # ATR-based stoploss: 2.5 * ATR below entry (wider for 6h volatility)
            stop_loss = entry_price - 2.5 * atr[i]
            # Exit if stoploss hit or Alligator loses bullish alignment
            if price < stop_loss or not (jaw[i] < teeth[i] < lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or Alligator reversal
            # ATR-based stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * atr[i]
            # Exit if stoploss hit or Alligator loses bearish alignment
            if price > stop_loss or not (jaw[i] > teeth[i] > lips[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals