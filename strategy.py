#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above R1 in uptrend (price > 1d EMA34), short when breaks below S1 in downtrend.
# Volume > 2x 20-period average confirms breakout strength. Avoids false breakouts in low volume.
# Target: 20-40 trades/year by requiring strict alignment of price, trend, and volume.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Pre-compute volume moving average (20-period)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        # Need previous day's OHLC (available at current 4h bar's open time)
        if i < 1:
            continue
            
        # Get previous day's data using 1d dataframe
        # Find the 1d bar that corresponds to yesterday
        current_date = df_1d.index[-1] if len(df_1d.index) > 0 else None
        # Simpler: use the last completed 1d bar available
        # We'll use the 1d bar that ended before current 4h bar
        
        # Calculate Camarilla from previous completed day
        # For simplicity, use rolling window on 4h data to approximate daily OHLC
        # But better: use actual 1d data from df_1d
        
        # Get the index of the last completed 1d bar
        # Since we're in 4h timeframe, we can use the 1d data directly
        # The aligned arrays give us values for each 4h bar
        
        # Actually, let's compute Camarilla using 1d OHLC from df_1d
        # We need the previous day's OHLC
        if len(df_1d) < 2:
            continue
            
        # For each 4h bar, use the most recent completed 1d bar's OHLC
        # This is already handled by alignment - we'll compute Camarilla on 1d then align
        
        # Calculate Camarilla levels on 1d data
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Typical price for Camarilla calculation
        typical_price = (high_1d + low_1d + close_1d) / 3
        
        # Camarilla levels
        R1 = close_1d + 1.1 * (high_1d - low_1d) / 12
        S1 = close_1d - 1.1 * (high_1d - low_1d) / 12
        R2 = close_1d + 1.1 * (high_1d - low_1d) / 6
        S2 = close_1d - 1.1 * (high_1d - low_1d) / 6
        
        # Align Camarilla levels to 4h timeframe
        R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
        S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
        
        # Current price and volume
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirm = volume > 2.0 * vol_ma[i]
        
        # Trend filter: price vs 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        if position == 0:
            if volume_confirm:
                # Long: price breaks above R1 in uptrend
                if uptrend and price > R1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: price breaks below S1 in downtrend
                elif downtrend and price < S1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit if price breaks below S1 (failed breakout) or trend changes
                if price < S1_aligned[i] or not uptrend:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit if price breaks above R1 (failed breakdown) or trend changes
                if price > R1_aligned[i] or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0