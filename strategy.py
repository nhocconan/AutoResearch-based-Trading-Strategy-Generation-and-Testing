#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H4/L4 breakout with 1w EMA50 trend filter and volume spike confirmation
# Camarilla H4/L4 levels act as stronger support/resistance than H3/L3 on 1d timeframe; breakouts with volume and 1w EMA50 trend alignment capture significant momentum moves
# Designed for ~12-37 trades/year to minimize fee drag while participating in established trends
# Works in bull/bear via 1w EMA50 trend filter - only trades in direction of 1w momentum
# Uses strict volume confirmation (>2.5x 20-period average) to reduce false breakouts and overtrading
# Exits on 2.0x ATR stoploss or when price retests the broken level (H4/L4)

name = "12h_Camarilla_H4L4_Breakout_1wEMA50_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values  # needed for Camarilla calculation
    
    # Get 1w data for EMA50 trend filter (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
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
    
    start_idx = 20  # volume MA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_1w = ema_50_1w_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla levels for this 12h bar using previous bar's OHLC
        if i == 0:
            signals[i] = 0.0
            continue
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_open = open_price[i-1]
        
        # Camarilla levels (based on previous bar's range)
        # H4/L4 are the key levels for breakout trading (stronger than H3/L3)
        H4 = prev_close + (prev_high - prev_low) * 1.1 / 2
        L4 = prev_close - (prev_high - prev_low) * 1.1 / 2
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below L4 (failed breakout)
            if curr_close < entry_price - 2.0 * curr_atr or curr_close < L4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above H4 (failed breakout)
            if curr_close > entry_price + 2.0 * curr_atr or curr_close > H4:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 2.5x 20-period average
            vol_confirm = curr_volume > 2.5 * curr_vol_ma
            
            # Long breakout when price closes above H4 with 1w EMA50 uptrend and volume confirmation
            if curr_close > H4 and curr_close > curr_ema50_1w and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short breakout when price closes below L4 with 1w EMA50 downtrend and volume confirmation
            elif curr_close < L4 and curr_close < curr_ema50_1w and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals