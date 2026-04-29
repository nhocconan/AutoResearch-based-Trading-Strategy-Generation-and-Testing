#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA34 trend filter and volume confirmation
# Uses Alligator jaws/teeth/lips for trend identification and potential reversals
# 1w EMA34 ensures alignment with weekly trend for higher probability trades
# Volume confirmation > 1.5x average to filter weak signals
# Discrete position sizing (0.25) with ATR-based stoploss for risk management
# Designed to capture trend reversals in both bull and bear markets with low frequency

name = "1d_WilliamsAlligator_1wEMA34_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Williams Alligator components (13,8,5 periods shifted)
    # Jaws: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(source, length):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < length:
            return result
        # First value is simple SMA
        result[length-1] = np.mean(source[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
        for i in range(length, len(source)):
            result[i] = (result[i-1] * (length-1) + source[i]) / length
        return result
    
    jaws = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (jaws 8, teeth 5, lips 3)
    jaws_shifted = np.roll(jaws, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Set NaN for shifted values that would look ahead
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Calculate ATR for volatility and stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 34, 13)  # Warmup for volume, 1w EMA, and Alligator
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_jaw = jaws_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        curr_ema34_1w = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: price below lips (Alligator waking up) OR ATR stoploss
            if curr_close < curr_lips or curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price above lips (Alligator waking up) OR ATR stoploss
            if curr_close > curr_lips or curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Alligator alignment: lips > teeth > jaws = uptrend, lips < teeth < jaws = downtrend
            # Volume confirmation: current volume > 1.5x 20-period average
            vol_confirmed = curr_volume > 1.5 * curr_vol_ma
            
            # Long when Alligator aligned for uptrend, price above lips, weekly trend up, volume confirmed
            if curr_lips > curr_teeth > curr_jaw and curr_close > curr_lips and curr_close > curr_ema34_1w and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when Alligator aligned for downtrend, price below lips, weekly trend down, volume confirmed
            elif curr_lips < curr_teeth < curr_jaw and curr_close < curr_lips and curr_close < curr_ema34_1w and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals