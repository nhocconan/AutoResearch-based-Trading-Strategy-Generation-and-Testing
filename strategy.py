#!/usr/bin/env python3
"""
4H_Bollinger_Upper_Band_Breakout_RSI_Exit
Hypothesis: Bollinger Band upper band breakouts with RSI > 60 capture momentum in trending markets.
Exits when RSI drops below 40. Designed for moderate frequency (~25-35 trades/year) to balance
capture of trends and minimize fee drag. Works in bull markets via breakouts and in bear
markets via short signals on lower band breaks with RSI < 40.
"""

name = "4H_Bollinger_Upper_Band_Breakout_RSI_Exit"
timeframe = "4h"
leverage = 1.0

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
    
    # === Bollinger Bands (20, 2) ===
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (bb_std * std)
    lower_band = sma - (bb_std * std)
    
    # === RSI(14) ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Volume Spike Filter (20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for Bollinger Bands and RSI)
    start_idx = bb_period  # 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_upper = close[i] > upper_band[i]
        price_below_lower = close[i] < lower_band[i]
        rsi_overbought = rsi[i] > 60
        rsi_oversold = rsi[i] < 40
        
        if position == 0:
            # Long: Price breaks above upper Bollinger Band + RSI > 60 + volume spike
            if price_above_upper and rsi_overbought and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below lower Bollinger Band + RSI < 40 + volume spike
            elif price_below_lower and rsi_oversold and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: RSI drops below 40
                if rsi[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit short: RSI rises above 60
                if rsi[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals