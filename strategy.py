# 4H TREND FOLLOWING WITH VOLUME CONFIRMATION AND RISK MANAGEMENT
# Hypothesis: On 4h timeframe, price tends to trend in the direction of the 4h EMA21 with momentum confirmed by RSI.
# Volume spikes confirm institutional participation. Trades only during high-volume periods to avoid chop.
# Works in both bull and bear markets by following the trend with proper risk management.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h data for EMA trend and RSI momentum ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # 4h EMA21 for trend direction
    ema21_4h = pd.Series(close_4h).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # 4h RSI(14) for momentum
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_4h = 100 - (100 / (1 + rs))
    rsi_4h = rsi_4h.values
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 4h volume average (20-period) for volume confirmation
    vol_avg20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_avg20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg20_4h)
    
    signals = np.zeros(n)
    
    # Warmup: covers EMA21, RSI, and rollouts
    warmup = 40
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(vol_avg20_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current 4h volume
        vol_4h_current = align_htf_to_ltf(prices, df_4h, volume_4h)[i]
        
        # Volume filter: current volume > 1.5x 20-period average
        vol_filter = vol_4h_current > 1.5 * vol_avg20_4h_aligned[i]
        
        # Trend following conditions
        if position == 0:
            # Long: price above EMA21, bullish momentum (RSI > 50), volume confirmation
            price_above_ema = close[i] > ema21_4h_aligned[i]
            bullish_momentum = rsi_4h_aligned[i] > 50
            
            if price_above_ema and bullish_momentum and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price below EMA21, bearish momentum (RSI < 50), volume confirmation
            price_below_ema = close[i] < ema21_4h_aligned[i]
            bearish_momentum = rsi_4h_aligned[i] < 50
            
            if price_below_ema and bearish_momentum and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: trend reversal or loss of momentum
        elif position == 1:
            # Exit long when price crosses below EMA21 or momentum fades
            price_below_ema = close[i] < ema21_4h_aligned[i]
            bearish_momentum = rsi_4h_aligned[i] < 40
            
            if price_below_ema or bearish_momentum:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price crosses above EMA21 or momentum fades
            price_above_ema = close[i] > ema21_4h_aligned[i]
            bullish_momentum = rsi_4h_aligned[i] > 60
            
            if price_above_ema or bullish_momentum:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_EMA21_RSI_Volume_Trend_Following"
timeframe = "4h"
leverage = 1.0