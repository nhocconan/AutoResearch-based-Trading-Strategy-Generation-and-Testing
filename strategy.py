#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams Alligator with 1w EMA50 trend filter and volume confirmation
# Uses Williams Alligator (jaw/teeth/lips) to identify trendless markets and avoid false signals
# 1w EMA50 as strong trend filter to only trade in direction of weekly trend
# Volume > 2.0x average confirms institutional participation to reduce false signals
# ATR-based stoploss (2.0x ATR) manages risk in volatile markets
# Discrete position sizing (0.25) to minimize fee churn
# Designed for ~10-25 trades/year to avoid fee drag while capturing strong trends
# Works in bull/bear via 1w EMA50 trend filter - only trades in direction of weekly trend
# Target: BTC/ETH focus with proven Williams Alligator structure + volume confirmation edge

name = "1d_WilliamsAlligator_1wEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Williams Alligator on 1d data (smoothed with 5-period SMA)
    # Jaw: 13-period SMA, shifted 8 bars into future
    # Teeth: 8-period SMA, shifted 5 bars into future  
    # Lips: 5-period SMA, shifted 3 bars into future
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Calculate ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = max(13+8, 8+5, 5+3, 50, 14, 20)  # Alligator, EMA, ATR, volume warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or Alligator lines cross (teeth below lips)
            if curr_close < entry_price - 2.0 * atr_at_entry or curr_teeth < curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or Alligator lines cross (teeth above lips)
            if curr_close > entry_price + 2.0 * atr_at_entry or curr_teeth > curr_lips:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Alligator alignment: lips > teeth > jaw = uptrend, lips < teeth < jaw = downtrend
            alligator_long = curr_lips > curr_teeth and curr_teeth > curr_jaw
            alligator_short = curr_lips < curr_teeth and curr_teeth < curr_jaw
            
            # Long when Alligator shows uptrend with 1w EMA50 uptrend and volume confirmation
            if alligator_long and curr_close > curr_ema50_1w and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short when Alligator shows downtrend with 1w EMA50 downtrend and volume confirmation
            elif alligator_short and curr_close < curr_ema50_1w and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals