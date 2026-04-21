#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above R1 with 12h EMA up and volume > 1.5x average.
# Short when price breaks below S1 with 12h EMA down and volume > 1.5x average.
# Exit on opposite Camarilla level (S1 for long, R1 for short) or trend reversal.
# Target: 20-40 trades/year by requiring strict alignment of price, trend, and volume.
# Works in bull/bear: breaks capture momentum, trend filter avoids counter-trend whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA trend (using close) - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 20-period volume moving average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        # Need previous day's high, low, close
        prev_day_start = max(0, i - 96)  # Approximate 1 day back (96 * 15min = 24h, but we're on 4h)
        # Better: use actual daily data from 1d timeframe
        if i >= 96:  # Ensure we have at least 1 day of 4h data
            # Get the previous day's OHLC from 4h data (simplified)
            # In practice, we'd use 1d data, but for simplicity we approximate
            day_high = prices['high'].iloc[i-24:i].max()  # Approximately 1 day back
            day_low = prices['low'].iloc[i-24:i].min()
            day_close = prices['close'].iloc[i-24:i].iloc[-1]
        else:
            # Not enough data for previous day
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Calculate Camarilla levels
        range_val = day_high - day_low
        if range_val <= 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        camarilla_r1 = day_close + (range_val * 1.1 / 12)
        camarilla_s1 = day_close - (range_val * 1.1 / 12)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume > 1.5 * vol_ma[i]
        
        # Trend filter: 12h EMA direction
        ema_rising = ema_12h_aligned[i] > ema_12h_aligned[i-1]
        ema_falling = ema_12h_aligned[i] < ema_12h_aligned[i-1]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above R1 with rising 12h EMA
                if price > camarilla_r1 and ema_rising:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 with falling 12h EMA
                elif price < camarilla_s1 and ema_falling:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price drops below S1 or 12h EMA turns down
                if price < camarilla_s1 or not ema_rising:
                    exit_signal = True
            elif position == -1:  # short position
                # Exit if price rises above R1 or 12h EMA turns up
                if price > camarilla_r1 or not ema_falling:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0