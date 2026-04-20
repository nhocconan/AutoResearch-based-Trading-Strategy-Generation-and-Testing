# 1d_PositionSize_VolatilityBreakout_v1
# Hypothesis: Breakouts above/below 1-day volatility bands with volume confirmation
# - Uses 1-day ATR to create adaptive volatility bands around the open
# - Entry: Price breaks above/below 1.5*ATR bands with volume > 1.5x 20-period average
# - Exit: Price returns to open level or time-based exit after 3 days
# - Volatility targeting ensures consistent risk regardless of market conditions
# - Works in both bull and bear markets by capturing volatility expansion moves
# - Target: 15-25 trades per year per symbol (60-100 total over 4 years)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate daily open price (using previous day's close as proxy for open)
    # For daily data, we'll use close as our primary price series
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ATR for volatility measurement
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate average true range for volatility bands
    atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    entry_bar = 0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if np.isnan(atr_ma[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Calculate volatility bands based on ATR
            upper_band = close[i-1] + 1.5 * atr_ma[i]
            lower_band = close[i-1] - 1.5 * atr_ma[i]
            
            # Long entry: price breaks above upper band with volume surge
            if price > upper_band and vol > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                entry_bar = i
            # Short entry: price breaks below lower band with volume surge
            elif price < lower_band and vol > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                entry_bar = i
        
        elif position != 0:
            # Exit conditions:
            # 1. Price returns to previous close (mean reversion)
            # 2. Time-based exit after 3 bars (3 days)
            # 3. Opposite band break (strong reversal)
            
            prev_close = close[i-1]
            
            if position == 1:  # Long position
                exit_condition = (price <= prev_close) or \
                                (i - entry_bar >= 3) or \
                                (price < close[i-1] - 1.5 * atr_ma[i])
            else:  # Short position
                exit_condition = (price >= prev_close) or \
                                (i - entry_bar >= 3) or \
                                (price > close[i-1] + 1.5 * atr_ma[i])
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_PositionSize_VolatilityBreakout_v1"
timeframe = "1d"
leverage = 1.0