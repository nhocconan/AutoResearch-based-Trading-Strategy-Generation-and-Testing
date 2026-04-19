#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaws (13-period SMA shifted 8) < teeth (8-period SMA shifted 5) < lips (5-period SMA shifted 3),
# price above lips, and 1d close > 1d EMA50, volume > 1.5x 20-period average.
# Short when jaws > teeth > lips, price below lips, and same filters.
# Uses discrete position size 0.25 to minimize churn. Designed for 4h to capture trends
# while avoiding whipsaws in both bull and bear markets via Alligator alignment and 1d trend filter.
# Target: 20-50 trades/year per symbol (~80-200 total over 4 years).
name = "4h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Williams Alligator components on 4h
    # Jaws: 13-period SMMA shifted 8 bars forward
    # Teeth: 8-period SMMA shifted 5 bars forward
    # Lips: 5-period SMMA shifted 3 bars forward
    # Using SMA for simplicity (close to SMMA for trend identification)
    jaws = pd.Series(close).rolling(window=13, min_periods=13).mean().shift(8).values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5).values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3).values
    
    # Align 1d EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13+8, 8+5, 5+3, 50, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        jaw_val = jaws[i]
        tooth_val = teeth[i]
        lip_val = lips[i]
        ema_50_val = ema_50_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        # 1d trend filter
        uptrend_1d = close_1d[-1] > ema_50_1d[-1] if len(close_1d) > 0 else False  # Simplified: use last known 1d value
        # Better: use the aligned EMA50 value to determine trend at this 4h bar
        # Since we can't access future 1d data, we use the fact that ema_50_aligned represents the 1d EMA50
        # value that was known at the close of the previous 1d bar
        # We'll determine trend by comparing close price to the 1d EMA50 (approximated)
        # Actually, we need to determine if we are in a 1d uptrend or downtrend
        # We can use the slope of the 1d EMA50, but simpler: use the 1d close vs EMA50 from the last completed 1d bar
        # For now, we'll use a simplified approach: if the current price is above the 1d EMA50 (aligned), consider uptrend
        # This is not perfect but avoids lookahead
        # Better approach: we'll determine 1d trend based on whether the 1d EMA50 is rising or falling
        # Since we can't do that easily without lookahead, we'll use price vs EMA50 as a proxy
        # Actually, let's use the aligned EMA50 value to get the 1d EMA50 value at this point in time
        # and compare it to what it was earlier to determine slope - but that's complex
        # Simpler and effective: use the 1d EMA50 value as dynamic support/resistance
        # Uptrend when price > 1d EMA50, downtrend when price < 1d EMA50
        uptrend_1d = price > ema_50_val
        downtrend_1d = price < ema_50_val
        
        # Alligator alignment: jaws < teeth < lips = uptrend alignment
        # jaws > teeth > lips = downtrend alignment
        alligator_long = jaw_val < tooth_val < lip_val
        alligator_short = jaw_val > tooth_val > lip_val
        
        if position == 0:
            # Enter long if Alligator aligned for uptrend, price above lips, 1d uptrend, and volume confirmation
            if alligator_long and price > lip_val and uptrend_1d and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Enter short if Alligator aligned for downtrend, price below lips, 1d downtrend, and volume confirmation
            elif alligator_short and price < lip_val and downtrend_1d and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when Alligator alignment breaks (jaws > teeth) or price crosses below lips
            if jaw_val > tooth_val or price < lip_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when Alligator alignment breaks (jaws < teeth) or price crosses above lips
            if jaw_val < tooth_val or price > lip_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals