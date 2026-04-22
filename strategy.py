# 4h_PriceAction_Reversal - Looking for rejection of daily extremes with volume confirmation
# Works in bull/bear markets by fading extreme price action at key daily levels
# Uses 1d high/low rejection with 4h price action and volume confirmation
# Designed for 4-8 trades per month per symbol (48-96/year) to avoid fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for key levels
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily high and low
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Align daily levels to 4h timeframe (wait for daily close)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low)
    
    # Calculate 4h ATR for volatility filtering
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Price and volume
    close_prices = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(daily_high_aligned[i]) or 
            np.isnan(daily_low_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = daily_high_aligned[i]
        dl = daily_low_aligned[i]
        atr_val = atr[i]
        price = close_prices[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        
        # Volume filter: above average volume
        vol_filter = vol > vol_ma_val
        
        # Distance from daily levels in ATR units
        dist_to_high = (dh - price) / atr_val if atr_val > 0 else 0
        dist_to_low = (price - dl) / atr_val if atr_val > 0 else 0
        
        if position == 0:
            # Long setup: price near daily low, rejecting downward
            # Look for rejection candle: close > open and near low
            if i > 0:
                open_price = prices['open'].iloc[i]
                close_price = close_prices[i]
                is_bullish = close_price > open_price
                near_low = dist_to_low < 0.5  # Within 0.5 ATR of daily low
                
                if is_bullish and near_low and vol_filter:
                    signals[i] = 0.25
                    position = 1
            
            # Short setup: price near daily high, rejecting upward
            # Look for rejection candle: close < open and near high
            if i > 0:
                open_price = prices['open'].iloc[i]
                close_price = close_prices[i]
                is_bearish = close_price < open_price
                near_high = dist_to_high < 0.5  # Within 0.5 ATR of daily high
                
                if is_bearish and near_high and vol_filter:
                    signals[i] = -0.25
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # Long position
                # Exit if price reaches daily high or reverses
                if price >= dh or (i > 0 and close_prices[i] < prices['open'].iloc[i]):
                    exit_signal = True
            
            elif position == -1:  # Short position
                # Exit if price reaches daily low or reverses
                if price <= dl or (i > 0 and close_prices[i] > prices['open'].iloc[i]):
                    exit_signal = True
            
            # Also exit if volatility drops significantly
            if i >= 20:
                atr_ma = pd.Series(atr).rolling(window=20, min_periods=20).mean().values[i]
                if atr_val < 0.5 * atr_ma:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_PriceAction_Reversal_DailyLevels"
timeframe = "4h"
leverage = 1.0