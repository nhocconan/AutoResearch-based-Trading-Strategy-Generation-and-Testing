#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above Camarilla R4 AND price > 1d EMA34 AND volume > 1.5x 20-period average
# Short when price breaks below Camarilla S4 AND price < 1d EMA34 AND volume > 1.5x 20-period average
# Uses ATR-based trailing stop (1.5x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee drag
# Target: 20-30 trades/year on 4h timeframe (~80-120 total over 4 years)
# Uses tighter Camarilla levels (R4/S4) vs R3/S3 to reduce trade frequency and improve win rate
# Works in bull markets via long breakouts with 1d uptrend
# Works in bear markets via short breakdowns with 1d downtrend

name = "4h_Camarilla_R4_S4_Breakout_1dEMA34_VolumeConfirm_v1"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Camarilla levels from previous day (using daily data)
    # Camarilla: R4 = C + ((H-L)*1.1/2), S4 = C - ((H-L)*1.1/2)
    # We use previous day's OHLC to calculate today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla R4 and S4 levels
    camarilla_range = prev_high - prev_low
    camarilla_R4 = prev_close + (camarilla_range * 1.1 / 2)
    camarilla_S4 = prev_close - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = max(100, 50, 50)  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        curr_R4 = camarilla_R4_aligned[i]
        curr_S4 = camarilla_S4_aligned[i]
        
        # Skip if Camarilla levels are not available
        if np.isnan(curr_R4) or np.isnan(curr_S4):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 1.5 * ATR below highest high
            stop_price = highest_high_since_entry - 1.5 * curr_atr
            # Exit conditions: price below trailing stop
            if curr_close < stop_price:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 1.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 1.5 * curr_atr
            # Exit conditions: price above trailing stop
            if curr_close > stop_price:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R4 AND price > 1d EMA34 AND volume spike
            if curr_close > curr_R4 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Camarilla S4 AND price < 1d EMA34 AND volume spike
            elif curr_close < curr_S4 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals