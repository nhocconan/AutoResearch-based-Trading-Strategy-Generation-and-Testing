#!/usr/bin/env python3
"""
1d Camarilla Pivot H3L3 Breakout + 1w EMA34 Trend + Volume Spike
Hypothesis: Camarilla H3/L3 levels act as strong intraday support/resistance on 1d timeframe.
Breakouts above H3 or below L3 with volume confirmation and 1w EMA34 trend alignment capture
strong momentum moves. Works in bull via breakouts with trend, in bear via failed breakouts
and short signals. Weekly EMA34 filter ensures we trade with the higher timeframe trend.
Discrete sizing (0.25) limits fee drag. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR for stop loss (using 20 periods)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for ATR (20) and EMA34 (34)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_34_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_34_aligned[i]
        atr_value = atr[i]
        
        # Prior day's OHLC for Camarilla calculation (using 1d timeframe)
        # Since we're on 1d timeframe, we use previous completed day's data
        if i >= 2:
            # Previous day's high, low, close
            prev_high = high[i-1]
            prev_low = low[i-1]
            prev_close = close[i-1]
            
            # Camarilla levels calculation
            range_val = prev_high - prev_low
            if range_val > 0:
                # H3 = close + (high - low) * 1.1/4
                # L3 = close - (high - low) * 1.1/4
                camarilla_h3 = prev_close + range_val * 1.1 / 4
                camarilla_l3 = prev_close - range_val * 1.1 / 4
            else:
                camarilla_h3 = prev_close
                camarilla_l3 = prev_close
        else:
            camarilla_h3 = close[i]
            camarilla_l3 = close[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        # Breakout conditions: price breaks above H3 or below L3
        bullish_breakout = curr_close > camarilla_h3
        bearish_breakout = curr_close < camarilla_l3
        
        # Update tracking variables for trailing stop logic
        if position == 1:
            highest_since_entry = max(highest_since_entry, curr_high)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, curr_low)
        
        # Exit conditions: trailing stop or reverse breakout
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Trailing stop: exit if price drops 3.0*ATR from highest since entry
                if curr_close < highest_since_entry - 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close < camarilla_l3 or curr_close < ema_trend:
                    exit_signal = True
                    
            elif position == -1:
                # Trailing stop: exit if price rises 3.0*ATR from lowest since entry
                if curr_close > lowest_since_entry + 3.0 * atr_value:
                    exit_signal = True
                # Reverse breakout or trend rejection
                elif curr_close > camarilla_h3 or curr_close > ema_trend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                continue
        
        # Entry conditions: Camarilla breakout + trend alignment + volume spike
        if position == 0:
            # Long: break above H3 AND price above 1w EMA34
            long_condition = bullish_breakout and (curr_close > ema_trend) and volume_spike
            # Short: break below L3 AND price below 1w EMA34
            short_condition = bearish_breakout and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
            elif short_condition:
                signals[i] = -0.25
                position = -1
                highest_since_entry = curr_high
                lowest_since_entry = curr_low
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_H3L3_Breakout_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0