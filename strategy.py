#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Use 1d RSI mean reversion for direction (long when oversold, short when overbought)
# with 1h timeframe for entry timing and volume confirmation. Filter trades to 08-20 UTC
# to avoid low liquidity periods. This should work in both bull and bear markets by
# capturing mean reversion moves within larger trends. Target: 15-35 trades/year.
# RSI(14) < 30 = oversold (long), RSI(14) > 70 = overbought (short)
# Volume > 1.5x average confirms genuine interest in the move.
# Position size: 0.20 (20%) to manage drawdown.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = pd.to_datetime(prices['open_time'])
    
    # Pre-compute hour for session filter (08-20 UTC)
    hours = open_time.dt.hour.values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 14-period RSI on daily timeframe
    delta = pd.Series(df_1d['close']).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs)).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Calculate average volume (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > (vol_avg[i] * 1.5) if not np.isnan(vol_avg[i]) else False
        
        # RSI-based mean reversion signals
        rsi = rsi_1d_aligned[i]
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        
        if position == 0:
            # Long setup: RSI oversold + volume confirmation
            if rsi_oversold and vol_confirm:
                position = 1
                signals[i] = position_size
            # Short setup: RSI overbought + volume confirmation
            elif rsi_overbought and vol_confirm:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral (above 40)
            if rsi > 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: RSI returns to neutral (below 60)
            if rsi < 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_1dRSI_MeanReversion_Volume"
timeframe = "1h"
leverage = 1.0