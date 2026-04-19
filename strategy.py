#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d RSI filter and volume confirmation.
# Long when price breaks above 20-period Donchian high AND 1d RSI < 30 (oversold) AND volume > 1.5x 20-period average volume
# Short when price breaks below 20-period Donchian low AND 1d RSI > 70 (overbought) AND volume > 1.5x 20-period average volume
# Exit when price crosses back through the Donchian midpoint
# Uses Donchian for trend following structure, 1d RSI for contrarian timing, volume for confirmation.
# Target: 20-30 trades/year per symbol.
name = "4h_Donchian_RSI_Volume"
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
    
    # Get 1d RSI for contrarian filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate RSI(14) on daily close
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(100).values  # Fill NaN with 100 for no signal
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_values)
    
    # Get 20-period average volume for confirmation
    vol_ma_20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(donchian_mid[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        vol = volume[i]
        upper = high_roll[i]
        lower = low_roll[i]
        mid = donchian_mid[i]
        
        if position == 0:
            # Long entry: break above upper band + oversold RSI + volume spike
            if price > upper and rsi_val < 30 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short entry: break below lower band + overbought RSI + volume spike
            elif price < lower and rsi_val > 70 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below midpoint
            if price < mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above midpoint
            if price > mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals