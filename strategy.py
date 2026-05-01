#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d Williams %R regime filter.
# Elder Ray: Bull Power = High - EMA13(Close), Bear Power = Low - EMA13(Close).
# Long when Bull Power > 0 AND 1d Williams %R < -80 (oversold) AND volume > 1.5x 20-median.
# Short when Bear Power < 0 AND 1d Williams %R > -20 (overbought) AND volume > 1.5x 20-median.
# Uses ATR(10) trailing stop: exit long if price < highest_since_entry - 2.0*ATR, exit short if price > lowest_since_entry + 2.0*ATR.
# Discrete sizing 0.25 to minimize fee churn. Target: 12-37 trades/year on 6h.
# Elder Ray measures bull/bear strength relative to trend; Williams %R on 1d adds regime filter for mean reversion extremes.
# Works in bull markets (buy oversold pullbacks) and bear markets (sell overbought rallies).

name = "6h_ElderRay_WilliamsR_1dRegime_VolumeSpike_ATR_v1"
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
    
    # Calculate 1d EMA13 for Elder Ray (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate EMA13 on 1d close
    ema_13_1d = pd.Series(df_1d['close']).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Lookback period 14
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - df_1d['close'].values) / (highest_high_14 - lowest_low_14 + 1e-10)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate 13-period EMA for Elder Ray on 6h close
    ema_13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13_6h  # High - EMA13
    bear_power = low - ema_13_6h   # Low - EMA13
    
    # Calculate 10-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for EMA, Williams %R, volume, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(ema_13_6h[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        
        # Williams %R regime filter: <-80 oversold, >-20 overbought
        williams_r = williams_r_1d_aligned[i]
        oversold = williams_r < -80
        overbought = williams_r > -20
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (bullish momentum) AND oversold AND volume spike
            if bull_power[i] > 0 and oversold and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Bear Power < 0 (bearish momentum) AND overbought AND volume spike
            elif bear_power[i] < 0 and overbought and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Update highest high since entry
            if curr_close > highest_since_entry:
                highest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR bear power turns negative (momentum loss)
            stop_price = highest_since_entry - 2.0 * curr_atr
            if curr_close < stop_price or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR bull power turns positive (momentum loss)
            stop_price = lowest_since_entry + 2.0 * curr_atr
            if curr_close > stop_price or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals