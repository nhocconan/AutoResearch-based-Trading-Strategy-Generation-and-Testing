# #!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike and 1d RSI(14) filter.
# Long when price breaks above upper Donchian channel + volume spike + 1d RSI > 50
# Short when price breaks below lower Donchian channel + volume spike + 1d RSI < 50
# Exit when price crosses back through middle Donchian band (average of upper/lower) or volume drops below 80% of average.
# Works in bull (breakouts with volume) and bear (breakdowns with volume) markets.
# Target: 20-40 trades/year to avoid excessive fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 14-period RSI on daily timeframe
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    
    # Calculate Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    upper_dc = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_dc = (upper_dc + lower_dc) / 2
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(upper_dc[i]) or 
            np.isnan(lower_dc[i]) or 
            np.isnan(middle_dc[i]) or 
            np.isnan(rsi_1d[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = upper_dc[i]
        lower = lower_dc[i]
        middle = middle_dc[i]
        rsi = rsi_1d[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above upper DC + volume spike + RSI > 50
            if price > upper and vol_spike and rsi > 50:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower DC + volume spike + RSI < 50
            elif price < lower and vol_spike and rsi < 50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price crosses back through middle DC or volume dries up
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below middle DC or volume dries up
                if price < middle or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above middle DC or volume dries up
                if price > middle or vol < 0.8 * vol_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0