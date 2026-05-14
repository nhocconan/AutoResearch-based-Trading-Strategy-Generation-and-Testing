#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 12h EMA50 trend filter and volume spike confirmation
# Williams Alligator: Jaw (13-period SMMA, 8-bar offset), Teeth (8-period SMMA, 5-bar offset), Lips (5-period SMMA, 3-bar offset)
# Long: Lips > Teeth > Jaw (bullish alignment) AND price > 12h EMA50 AND volume > 1.8x 20-bar avg
# Short: Lips < Teeth < Jaw (bearish alignment) AND price < 12h EMA50 AND volume > 1.8x 20-bar avg
# Exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw) OR price crosses 12h EMA50 OR ATR stoploss (2.0 * ATR)
# Williams Alligator identifies trend presence and direction with less whipsaw than single MAs
# 12h EMA50 provides stronger trend filter than shorter HTF, reducing false signals in chop
# Volume spike confirms breakout strength and institutional participation
# Discrete position sizing: 0.25 for long/short to minimize fee churn while maintaining adequate exposure
# Target: 100-180 total trades over 4 years (25-45/year) on 4h timeframe

name = "4h_Williams_Alligator_12hEMA50_VolumeSpike_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Williams Alligator components using SMMA (Smoothed Moving Average)
    # SMMA is similar to EMA but with different smoothing factor
    def smma(source, period):
        if len(source) < period:
            return np.full_like(source, np.nan)
        result = np.full_like(source, np.nan)
        # First value is SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    # Jaw: 13-period SMMA of median price, 8 bars offset
    median_price = (high + low) / 2
    jaw_raw = smma(median_price, 13)
    jaw = np.roll(jaw_raw, 8)  # 8 bars offset into future
    
    # Teeth: 8-period SMMA of median price, 5 bars offset
    teeth_raw = smma(median_price, 8)
    teeth = np.roll(teeth_raw, 5)  # 5 bars offset into future
    
    # Lips: 5-period SMMA of median price, 3 bars offset
    lips_raw = smma(median_price, 5)
    lips = np.roll(lips_raw, 3)  # 3 bars offset into future
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 13+8)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any Alligator line is not yet calculated (NaN due to roll offset)
        if np.isnan(lips[i]) or np.isnan(teeth[i]) or np.isnan(jaw[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_ema_12h = ema_50_12h_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2 * ATR below entry
            stop_price = entry_price - 2.0 * curr_atr
            # Exit conditions: Alligator lines cross (Lips-Teeth or Teeth-Jaw) OR price below 12h EMA50 OR stoploss hit
            if (lips[i] < teeth[i] or teeth[i] < jaw[i]) or curr_close < curr_ema_12h or curr_close < stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2 * ATR above entry
            stop_price = entry_price + 2.0 * curr_atr
            # Exit conditions: Alligator lines cross (Lips-Teeth or Teeth-Jaw) OR price above 12h EMA50 OR stoploss hit
            if (lips[i] > teeth[i] or teeth[i] > jaw[i]) or curr_close > curr_ema_12h or curr_close > stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: Lips > Teeth > Jaw (bullish alignment) AND price > 12h EMA50 AND volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                curr_close > curr_ema_12h and
                vol_spike):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: Lips < Teeth < Jaw (bearish alignment) AND price < 12h EMA50 AND volume spike
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  curr_close < curr_ema_12h and
                  vol_spike):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals