#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h trend filter and volume confirmation
# Long when RSI < 30 (oversold) + price above 4h EMA20 (uptrend) + volume spike
# Short when RSI > 70 (overbought) + price below 4h EMA20 (downtrend) + volume spike
# Exit when RSI returns to neutral (40-60 range) or trend changes
# Uses 4h for trend direction (reduces false signals), 1h for precise entry timing
# Session filter (08-20 UTC) to avoid low-liquidity hours
# Target: 15-35 trades/year to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA20 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA20 for trend filter
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate 1h RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(ema20_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter (08-20 UTC)
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rsi_val = rsi[i]
        ema20 = ema20_4h_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0 and in_session:
            # Long conditions: RSI < 30 (oversold) + price > 4h EMA20 (uptrend) + volume spike
            if rsi_val < 30 and price > ema20 and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI > 70 (overbought) + price < 4h EMA20 (downtrend) + volume spike
            elif rsi_val > 70 and price < ema20 and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI returns to neutral (40-60) or trend changes
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI > 40 or price breaks below 4h EMA20
                if rsi_val > 40 or price < ema20:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI < 60 or price breaks above 4h EMA20
                if rsi_val < 60 or price > ema20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA20_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0