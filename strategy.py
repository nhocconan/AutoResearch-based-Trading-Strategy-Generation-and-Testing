#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume spike confirmation
# Camarilla H3/L3 levels act as strong support/resistance on 1h timeframe; breakouts with volume and 4h EMA50 trend alignment capture momentum moves
# Designed for ~15-37 trades/year to minimize fee drag while participating in established trends
# Works in bull/bear via 4h EMA50 trend filter - only trades in direction of 4h momentum
# Uses strict volume confirmation (>2.0x 20-period average) to reduce false breakouts and overtrading
# Exits on 1.5x ATR stoploss or when price retests the broken level (H3/L3)

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_VolumeSpike_v1"
timeframe = "1h"
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
    
    # Get 4h data for EMA50 trend filter (HTF = 4h)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
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
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_open = open_price[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_ema50_4h = ema_50_4h_aligned[i]
        curr_atr = atr[i]
        curr_vol_ma = vol_ma_20[i]
        
        # Calculate Camarilla levels for this 1h bar using previous bar's OHLC
        if i == 0:
            signals[i] = 0.0
            continue
        prev_close = close[i-1]
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_open = open_price[i-1]
        
        # Camarilla levels (based on previous bar's range)
        # H3/L3 are the key levels for breakout trading
        H3 = prev_close + (prev_high - prev_low) * 1.1 / 4
        L3 = prev_close - (prev_high - prev_low) * 1.1 / 4
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: stoploss hit or price breaks below L3 (failed breakout)
            if curr_close < entry_price - 1.5 * curr_atr or curr_close < L3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: stoploss hit or price breaks above H3 (failed breakout)
            if curr_close > entry_price + 1.5 * curr_atr or curr_close > H3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
                
        else:  # Flat - look for new breakout entries
            # Volume confirmation: current volume > 2.0x 20-period average
            vol_confirm = curr_volume > 2.0 * curr_vol_ma
            
            # Long breakout when price closes above H3 with 4h EMA50 uptrend and volume confirmation
            if curr_close > H3 and curr_close > curr_ema50_4h and vol_confirm:
                signals[i] = 0.20
                position = 1
                entry_price = curr_close
                atr_at_entry = curr_atr
            # Short breakout when price closes below L3 with 4h EMA50 downtrend and volume confirmation
            elif curr_close < L3 and curr_close < curr_ema50_4h and vol_confirm:
                signals[i] = -0.20
                position = -1
                entry_price = curr_close
                atr_at_entry = curr_atr
            else:
                signals[i] = 0.0
    
    return signals