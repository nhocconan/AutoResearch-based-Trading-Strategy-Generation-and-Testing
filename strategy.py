#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h ADX trend filter and 1d RSI mean reversion.
# Uses 4h ADX > 25 to identify trending markets (breakout on 1h breakouts with volume).
# Uses 1d RSI < 30 or > 70 to identify overextended conditions for mean reversion in ranging markets.
# Combines trend and mean reversion strategies based on market regime to work in both bull and bear markets.
# Targets 15-30 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data for ADX (trend filter) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX components on 4h data
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.where((high_4h - np.roll(high_4h, 1)) > (np.roll(low_4h, 1) - low_4h), 
                       np.maximum(high_4h - np.roll(high_4h, 1), 0), 0)
    minus_dm = np.where((np.roll(low_4h, 1) - low_4h) > (high_4h - np.roll(high_4h, 1)), 
                        np.maximum(np.roll(low_4h, 1) - low_4h, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smooth with Welles Wilder smoothing (equivalent to RMA)
    def rma(values, period):
        result = np.zeros_like(values)
        if len(values) >= period:
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    tr_rma = rma(tr, 14)
    plus_dm_rma = rma(plus_dm, 14)
    minus_dm_rma = rma(minus_dm, 14)
    
    # Avoid division by zero
    plus_di = np.where(tr_rma != 0, 100 * plus_dm_rma / tr_rma, 0)
    minus_di = np.where(tr_rma != 0, 100 * minus_dm_rma / tr_rma, 0)
    
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = rma(dx, 14)
    
    # Align ADX to 1h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Load 1d data for RSI (mean reversion signal) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI on 1d data
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # RSI smoothing with Wilder's method
    def rsi_rma(values, period):
        result = np.zeros_like(values)
        if len(values) >= period:
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    avg_gain = rsi_rma(gain, 14)
    avg_loss = rsi_rma(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 1h volume moving average for confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1h price range for breakout levels
    high = prices['high'].values
    low = prices['low'].values
    # Use 24-period high/low for 1-day equivalent on 1h chart
    high_24 = pd.Series(high).rolling(window=24, min_periods=24).max().values
    low_24 = pd.Series(low).rolling(window=24, min_periods=24).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or 
            np.isnan(high_24[i]) or 
            np.isnan(low_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        rsi_val = rsi_aligned[i]
        upper_24 = high_24[i]
        lower_24 = low_24[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine market regime using 4h ADX
            is_trending = adx_val > 25  # Strong trend
            
            if is_trending:
                # Trending regime: breakout strategy in direction of momentum
                if price > upper_24 and vol_spike:
                    signals[i] = 0.20
                    position = 1
                elif price < lower_24 and vol_spike:
                    signals[i] = -0.20
                    position = -1
            else:
                # Ranging regime: mean reversion at extremes
                if rsi_val < 30 and vol_spike:  # Oversold
                    signals[i] = 0.20
                    position = 1
                elif rsi_val > 70 and vol_spike:  # Overbought
                    signals[i] = -0.20
                    position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit on RSI overbought or price retracement to midpoint
                if rsi_val > 70 or price < (upper_24 + lower_24) / 2:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit on RSI oversold or price retracement to midpoint
                if rsi_val < 30 or price > (upper_24 + lower_24) / 2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_ADX_Trend_RSI_MeanRev"
timeframe = "1h"
leverage = 1.0