#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Long when Alligator jaws < teeth < lips (bullish alignment) AND price > 1d EMA34 AND volume > 2.0x 20-period average
# Short when Alligator jaws > teeth > lips (bearish alignment) AND price < 1d EMA34 AND volume > 2.0x 20-period average
# Williams Alligator uses smoothed median prices: jaws=13-period SMMA(8), teeth=8-period SMMA(5), lips=5-period SMMA(3)
# Works in bull markets via long entries with 1d uptrend and bullish Alligator alignment
# Works in bear markets via short entries with 1d downtrend and bearish Alligator alignment
# Volume confirmation ensures strong participation
# Target: 12-30 trades/year on 12h timeframe to avoid fee drag while capturing strong trends
# Discrete position sizing (0.25) to minimize fee churn

name = "12h_Williams_Alligator_1dEMA34_VolumeConfirm_v1"
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
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Median price = (high + low) / 2
    median_price = (high + low) / 2.0
    
    # Jaws: 13-period SMMA of median price, shifted 8 bars
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    # Lips: 5-period SMMA of median price, shifted 3 bars
    # SMMA (Smoothed Moving Average) = EMA with alpha = 1/period
    
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        # First value is simple average
        result = np.full_like(arr, np.nan)
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(median_price, 13)  # 13-period SMMA
    teeth = smma(median_price, 8)   # 8-period SMMA
    lips = smma(median_price, 5)    # 5-period SMMA
    
    # Apply shifts: jaws shifted 8, teeth shifted 5, lips shifted 3
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that rolled from end
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(50, 34)  # warmup for EMA and Alligator
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_jaw = jaws_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lip = lips_shifted[i]
        curr_atr = atr[i]
        
        # Skip if Alligator values are not available
        if np.isnan(curr_jaw) or np.isnan(curr_teeth) or np.isnan(curr_lip):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 2.0x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, high[i])
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, low[i])
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Bullish Alligator alignment: jaws < teeth < lips
            bullish_alignment = curr_jaw < curr_teeth < curr_lip
            # Bearish Alligator alignment: jaws > teeth > lips
            bearish_alignment = curr_jaw > curr_teeth > curr_lip
            
            # Long entry: bullish alignment AND price > 1d EMA34 AND volume spike
            if bullish_alignment and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short entry: bearish alignment AND price < 1d EMA34 AND volume spike
            elif bearish_alignment and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
            else:
                signals[i] = 0.0
    
    return signals