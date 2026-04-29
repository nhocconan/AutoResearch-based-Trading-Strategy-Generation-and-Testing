#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels (R3/S3) act as strong intraday support/resistance levels
# Long when price breaks above R3 AND price > 1d EMA34 AND volume > 2.0x 24-period average
# Short when price breaks below S3 AND price < 1d EMA34 AND volume > 2.0x 24-period average
# Uses ATR-based trailing stop (2.5x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee churn
# Target: 12-30 trades/year on 12h timeframe to avoid fee drag while capturing institutional levels
# Works in bull markets via long breakouts at R3 with HTF uptrend
# Works in bear markets via short breakdowns at S3 with HTF downtrend
# Volume confirmation ensures breakouts have conviction, reducing false signals
# Using 1d EMA34 provides smoother trend filter than 12h EMA50, reducing whipsaws in ranging markets

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_VolumeConfirm_v1"
timeframe = "12h"
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla pivot levels from previous day
        if i >= 24:  # Need at least 24 bars (previous day) for calculation
            # Use previous day's high, low, close (24 bars ago for 12h timeframe)
            prev_high = np.max(high[i-24:i])
            prev_low = np.min(low[i-24:i])
            prev_close = close[i-1]  # Previous bar's close
            
            # Camarilla levels
            range_val = prev_high - prev_low
            camarilla_r3 = prev_close + range_val * 1.1 / 4
            camarilla_s3 = prev_close - range_val * 1.1 / 4
        else:
            camarilla_r3 = curr_high
            camarilla_s3 = curr_low
        
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
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop OR price breaks below Camarilla S3
            if curr_close < stop_price or curr_close < camarilla_s3:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Update lowest low since entry
            lowest_low_since_entry = min(lowest_low_since_entry, curr_low)
            # Trailing stop: 2.5 * ATR above lowest low
            stop_price = lowest_low_since_entry + 2.5 * curr_atr
            # Exit conditions: price above trailing stop OR price breaks above Camarilla R3
            if curr_close > stop_price or curr_close > camarilla_r3:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Camarilla R3 AND price > 1d EMA34 AND volume spike
            if curr_close > camarilla_r3 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below Camarilla S3 AND price < 1d EMA34 AND volume spike
            elif curr_close < camarilla_s3 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals