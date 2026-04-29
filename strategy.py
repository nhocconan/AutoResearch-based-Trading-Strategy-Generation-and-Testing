#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation
# Camarilla pivot levels provide precise intraday support/resistance derived from prior day's range
# Long when price breaks above R1 AND price > 1d EMA34 AND volume > 1.8x 20-period average
# Short when price breaks below S1 AND price < 1d EMA34 AND volume > 1.8x 20-period average
# Uses ATR-based trailing stop (2.5x ATR) for risk management
# Discrete position sizing (0.25) to minimize fee churn
# Target: 30-50 trades/year on 4h timeframe to avoid fee drag while capturing institutional breakouts
# Works in bull markets via long R1 breakouts with HTF uptrend
# Works in bear markets via short S1 breakdowns with HTF downtrend
# Volume confirmation ensures breakouts have institutional participation, reducing false signals
# Using 1d EMA34 provides smoother trend filter than shorter EMAs, reducing whipsaws in chop

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate ATR for stoploss (using 15-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=15, adjust=False, min_periods=15).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    start_idx = 50  # warmup for EMA and ATR
    
    for i in range(start_idx, n):
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        # Calculate Camarilla pivot levels from previous day's range
        # Need previous day's high, low, close - use 1d data aligned to current 4h bar
        if i >= 6:  # Need at least 6 bars back (24h) for previous day
            # Get previous day's OHLC from 1d data
            # Find index in 1d data corresponding to previous completed day
            prev_day_idx = len(df_1d) - 1  # This is approximate; align_htf_to_ltf handles proper alignment
            # Instead, we'll use rolling window on 1d data but we need to access it properly
            # Simpler approach: use current 4h bar's index to get previous day's data
            pass  # We'll calculate Camarilla levels differently
        
        # Simpler Camarilla calculation: use rolling 24-period (1d in 4h) high/low/close
        if i >= 24:
            prev_high = np.max(high[i-24:i])
            prev_low = np.min(low[i-24:i])
            prev_close = close[i-1]  # Previous bar's close
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_val = prev_high - prev_low
            r1 = pivot + (range_val * 1.1 / 12)
            s1 = pivot - (range_val * 1.1 / 12)
        else:
            # Not enough data, use current bar approximations
            r1 = curr_high
            s1 = curr_low
        
        # Volume spike confirmation: current volume > 1.8x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.8 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Update highest high since entry
            highest_high_since_entry = max(highest_high_since_entry, curr_high)
            # Trailing stop: 2.5 * ATR below highest high
            stop_price = highest_high_since_entry - 2.5 * curr_atr
            # Exit conditions: price below trailing stop OR price breaks below S1
            if curr_close < stop_price or curr_close < s1:
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
            # Exit conditions: price above trailing stop OR price breaks above R1
            if curr_close > stop_price or curr_close > r1:
                signals[i] = 0.0
                position = 0
                lowest_low_since_entry = 0.0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above R1 AND price > 1d EMA34 AND volume spike
            if curr_close > r1 and curr_close > curr_ema_1d and vol_spike:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = curr_high
            # Short entry: price breaks below S1 AND price < 1d EMA34 AND volume spike
            elif curr_close < s1 and curr_close < curr_ema_1d and vol_spike:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = curr_low
            else:
                signals[i] = 0.0
    
    return signals