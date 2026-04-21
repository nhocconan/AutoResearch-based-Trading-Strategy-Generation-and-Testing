#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Chande Momentum Oscillator (CMO) reversal with 1d trend filter and volume confirmation.
# Long when CMO < -50 (oversold) in 1d uptrend (close > EMA50), short when CMO > 50 (overbought) in 1d downtrend (close < EMA50).
# Volume > 1.3x 20-period average confirms momentum exhaustion. Uses EMA50 for trend to avoid counter-trend trades.
# Target: 20-40 trades/year by requiring overextended momentum + trend alignment + volume confirmation.
# Works in bull/bear: EMA50 filter ensures trades align with higher timeframe trend, reducing whipsaws.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6-period Chande Momentum Oscillator (CMO) on 6h data
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    def Wilder_smooth(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    sum_up = Wilder_smooth(up, 6)
    sum_down = Wilder_smooth(down, 6)
    cmo = 100 * (sum_up - sum_down) / (sum_up + sum_down)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(cmo[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume > 1.3 * vol_ma[i]
        
        # Trend filter: 1d close relative to EMA50
        uptrend = price > ema50_1d_aligned[i]
        downtrend = price < ema50_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: oversold in uptrend
                if cmo[i] < -50 and uptrend:
                    signals[i] = 0.25
                    position = 1
                # Short: overbought in downtrend
                elif cmo[i] > 50 and downtrend:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if CMO returns to neutral or trend breaks
                if cmo[i] >= -10 or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if CMO returns to neutral or trend breaks
                if cmo[i] <= 10 or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_CMO_OversoldOverbought_1dEMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0