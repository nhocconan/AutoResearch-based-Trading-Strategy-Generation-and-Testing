#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA50 trend filter and volume confirmation.
# Williams %R measures overbought/oversold levels: > -20 = overbought, < -80 = oversold.
# Long when Williams %R(14) crosses above -80 from below AND price > 1d EMA50 AND volume > 1.5x 20-period median.
# Short when Williams %R(14) crosses below -20 from above AND price < 1d EMA50 AND volume > 1.5x 20-period median.
# Uses ATR(14) stoploss: exit long if price < highest_since_entry - 2.0*ATR(14), exit short if price > lowest_since_entry + 2.0*ATR(14).
# Williams %R is effective in ranging markets and catches reversals at extremes, working in both bull and bear regimes.
# The 1d EMA50 filter ensures we trade with the higher timeframe trend, reducing counter-trend whipsaws.
# Volume confirmation adds conviction to reversals. Discrete sizing (0.25) minimizes fee churn.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).

name = "6h_WilliamsR_Extreme_1dEMA50_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA50 for trend filter (loaded once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1d close
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 14-period ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start after warmup for Williams %R, EMA, volume, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        curr_williams_r = williams_r[i]
        prev_williams_r = williams_r[i-1] if i > 0 else -50
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Williams %R signals: cross above -80 (oversold) or cross below -20 (overbought)
        williams_long_signal = (prev_williams_r < -80) and (curr_williams_r >= -80)
        williams_short_signal = (prev_williams_r > -20) and (curr_williams_r <= -20)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R crosses above -80 AND uptrend AND volume confirmation
            if williams_long_signal and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Williams %R crosses below -20 AND downtrend AND volume confirmation
            elif williams_short_signal and downtrend and volume_confirm:
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
            
            # Exit conditions: ATR stoploss OR Williams %R crosses above -20 (overbought) OR trend reversal
            stop_price = highest_since_entry - 2.0 * curr_atr
            williams_exit_long = (prev_williams_r < -20) and (curr_williams_r >= -20)
            if curr_close < stop_price or williams_exit_long or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Williams %R crosses below -80 (oversold) OR trend reversal
            stop_price = lowest_since_entry + 2.0 * curr_atr
            williams_exit_short = (prev_williams_r > -80) and (curr_williams_r <= -80)
            if curr_close > stop_price or williams_exit_short or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals