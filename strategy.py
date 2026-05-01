#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above upper BB(20,2) AND 1d EMA50 uptrend AND volume > 1.5x 20-period median.
# Short when price breaks below lower BB(20,2) AND 1d EMA50 downtrend AND volume > 1.5x 20-period median.
# Uses ATR-based trailing stop: exit if price moves against position by 2.5*ATR(14) from favorable extreme.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to avoid fee drag.
# Bollinger Bands adapt to volatility, providing dynamic support/resistance that works in both bull and bear markets.
# Volume confirmation reduces false breakouts. EMA50 filter ensures trades align with higher-timeframe trend.

name = "6h_Bollinger_Breakout_1dEMA50_Volume_ATR_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    favorable_extreme = 0.0  # highest high for long, lowest low for short
    
    # Start after warmup for EMA, BB, volume, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or 
            np.isnan(upper_bb[i]) or 
            np.isnan(lower_bb[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above upper BB AND uptrend AND volume spike
            if curr_close > upper_bb[i] and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                favorable_extreme = curr_close
            # Short: Price breaks below lower BB AND downtrend AND volume spike
            elif curr_close < lower_bb[i] and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                favorable_extreme = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update favorable extreme (highest high since entry)
            if curr_close > favorable_extreme:
                favorable_extreme = curr_close
            
            # Exit conditions: ATR trailing stop OR BB mean reversion OR trend reversal
            stop_price = favorable_extreme - 2.5 * curr_atr
            if curr_close < stop_price or curr_close < sma_20[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update favorable extreme (lowest low since entry)
            if curr_close < favorable_extreme:
                favorable_extreme = curr_close
            
            # Exit conditions: ATR trailing stop OR BB mean reversion OR trend reversal
            stop_price = favorable_extreme + 2.5 * curr_atr
            if curr_close > stop_price or curr_close > sma_20[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals