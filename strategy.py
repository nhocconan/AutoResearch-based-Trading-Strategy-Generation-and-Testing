#!/usr/bin/env python3
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
    
    # Load daily data for 1d ATR calculation - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 14:
        return np.zeros(n)
    
    # Calculate 14-day ATR (True Range based)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # True Range
    tr1 = high_daily - low_daily
    tr2 = np.abs(high_daily - np.roll(close_daily, 1))
    tr3 = np.abs(low_daily - np.roll(close_daily, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR calculation
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_daily, atr_daily)
    
    # Load 4h data for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4-period RSI on 4h closes (for momentum filter)
    delta = np.diff(close_4h, prepend=close_4h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=4, min_periods=4).mean().values
    avg_loss = pd.Series(loss).rolling(window=4, min_periods=4).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_4h = 100 - (100 / (1 + rs))
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(rsi_4h[i]) or 
            np.isnan(high_4h[i]) or np.isnan(low_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate 4h ATR for entry triggers
        tr_4h = np.maximum(high_4h - low_4h, 
                          np.maximum(np.abs(high_4h - np.roll(close_4h, 1)), 
                                    np.abs(low_4h - np.roll(close_4h, 1))))
        tr_4h[0] = high_4h[0] - low_4h[0]
        atr_4h_series = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean()
        atr_4h = atr_4h_series.values
        
        if position == 0:
            # Long: Bullish engulfing + RSI > 50 + price above 4h EMA(20) with volume
            bullish_engulfing = (close_4h[i] > open_4h[i] and 
                               open_4h[i] > close_4h[i-1] and 
                               close_4h[i] > close_4h[i-1])
            ema_20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20).mean().values
            
            if (bullish_engulfing and 
                rsi_4h[i] > 50 and 
                close_4h[i] > ema_20_4h[i] and 
                volume[i] > 1.5 * pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish engulfing + RSI < 50 + price below 4h EMA(20) with volume
            elif (close_4h[i] < open_4h[i] and 
                  open_4h[i] < close_4h[i-1] and 
                  close_4h[i] < close_4h[i-1] and
                  rsi_4h[i] < 50 and 
                  close_4h[i] < ema_20_4h[i] and 
                  volume[i] > 1.5 * pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Dynamic exit: ATR-based trailing stop
            if position == 1:
                # Trail long: exit if price drops below highest high since entry - 2*ATR
                if i == 1:
                    highest_since_entry = high_4h[i]
                else:
                    highest_since_entry = max(highest_since_entry, high_4h[i])
                
                if close_4h[i] < highest_since_entry - 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Trail short: exit if price rises above lowest low since entry + 2*ATR
                if i == 1:
                    lowest_since_entry = low_4h[i]
                else:
                    lowest_since_entry = min(lowest_since_entry, low_4h[i])
                
                if close_4h[i] > lowest_since_entry + 2.0 * atr_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

# Strategy: 4H_BullishEngulfing_RSI_EMA20_ATRTrail
# Hypothesis: 4-hour bullish/bearish engulfing candles with RSI momentum filter
# and EMA(20) trend filter, using ATR-based trailing stops for risk management.
# Works in bull/bear markets by focusing on price action patterns rather than
# directional bias. Session filter (08-20 UTC) avoids low-liquidity periods.
# Target: 20-50 trades/year (80-200 total over 4 years).

name = "4H_BullishEngulfing_RSI_EMA20_ATRTrail"
timeframe = "4h"
leverage = 1.0