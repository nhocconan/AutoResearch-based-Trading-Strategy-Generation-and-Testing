#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R (14) + 1d EMA50 trend filter + volume confirmation.
# Williams %R identifies overbought/oversold conditions; EMA50 determines trend direction.
# In uptrend (price > EMA50), buy when Williams %R crosses above -50 from below (bullish momentum).
# In downtrend (price < EMA50), sell when Williams %R crosses below -50 from above (bearish momentum).
# Volume spike confirms momentum strength. Designed for 6h timeframe to capture medium-term swings.
# Targets 15-35 trades/year with controlled risk.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA50 trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R (14) on 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate highest high and lowest low over 14 periods
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Williams %R = -100 * (HH - Close) / (HH - LL)
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)  # Avoid division by zero
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema50 = ema50_aligned[i]
        wr = williams_r[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Uptrend: price > EMA50
            if price > ema50:
                # Buy when Williams %R crosses above -50 from below (bullish momentum)
                if i > 0 and wr > -50 and williams_r[i-1] <= -50 and vol_spike:
                    signals[i] = 0.25
                    position = 1
            # Downtrend: price < EMA50
            elif price < ema50:
                # Sell when Williams %R crosses below -50 from above (bearish momentum)
                if i > 0 and wr < -50 and williams_r[i-1] >= -50 and vol_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when Williams %R crosses below -80 (overbought) or trend changes
                if wr < -80 or price < ema50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when Williams %R crosses above -20 (oversold) or trend changes
                if wr > -20 or price > ema50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_EMA50_Trend_Volume"
timeframe = "6h"
leverage = 1.0