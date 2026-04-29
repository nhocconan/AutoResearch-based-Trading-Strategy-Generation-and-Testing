#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla R3/S3 levels represent stronger intraday support/resistance than R1/S1
# Long when price breaks above R3 AND price > 1d EMA34 AND volume > 2.0x 24-period average
# Short when price breaks below S3 AND price < 1d EMA34 AND volume > 2.0x 24-period average
# Uses ATR-based trailing stop (2.0x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee churn
# Target: 20-50 trades/year on 4h timeframe to avoid fee drag while capturing strong breakouts
# Works in bull markets via long R3 breakouts with 1d uptrend
# Works in bear markets via short S3 breakdowns with 1d downtrend
# Volume confirmation with higher threshold ensures breakouts have strong institutional participation

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for stoploss (using 20-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 34  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla pivot levels from previous day's range
        # Need previous day's high, low, close - use 1d data
        if i >= 1:  # Need at least 1 bar back for previous day in 4h timeframe (6 bars = 1 day)
            # Get previous day's OHLC from 1d data aligned to current 4h bar
            high_1d = df_1d['high'].values
            low_1d = df_1d['low'].values
            close_1d = df_1d['close'].values
            
            # Get previous day's OHLC (6 bars back in 4h = 1 day back)
            if len(high_1d) >= 2:
                prev_high_1d = high_1d[-2]
                prev_low_1d = low_1d[-2]
                prev_close_1d = close_1d[-2]
            else:
                prev_high_1d = high_1d[-1]
                prev_low_1d = low_1d[-1]
                prev_close_1d = close_1d[-1]
            
            # Calculate Camarilla levels
            pivot = (prev_high_1d + prev_low_1d + prev_close_1d) / 3.0
            range_val = prev_high_1d - prev_low_1d
            r3 = pivot + (range_val * 1.1 / 4)  # R3 level
            s3 = pivot - (range_val * 1.1 / 4)  # S3 level
        else:
            # Not enough data, use current bar approximations
            r3 = curr_high
            s3 = curr_low
        
        # Volume spike confirmation: current volume > 2.0x 24-period average
        if i >= 24:
            vol_ma_24 = np.mean(volume[i-24:i])
        else:
            vol_ma_24 = 0.0
        vol_spike = volume[i] > 2.0 * vol_ma_24 if vol_ma_24 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.0 * ATR below highest high
            stop_price = highest_high_since_entry - 2.0 * curr_atr
            # Exit conditions: price below trailing stop OR price breaks below S3
            if curr_close < stop_price or curr_close < s3:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.0 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.0 * curr_atr
            # Exit conditions: price above trailing stop OR price breaks above R3
            if curr_close > stop_price or curr_close > r3:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R3 AND price > 1d EMA34 AND volume spike
            if curr_close > r3 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below S3 AND price < 1d EMA34 AND volume spike
            elif curr_close < s3 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals