#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d Williams %R regime filter and volume confirmation.
# Long when Bull Power > 0 AND 1d Williams %R < -80 (oversold) AND volume > 1.5x 20-period median.
# Short when Bear Power < 0 AND 1d Williams %R > -20 (overbought) AND volume > 1.5x 20-period median.
# Uses ATR(14) trailing stop: exit long if price < highest_since_entry - 2.5*ATR(14), exit short if price > lowest_since_entry + 2.5*ATR(14).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 15-30 trades/year on 6h timeframe.
# Elder Ray measures bull/bear strength relative to EMA13. Williams %R on 1d identifies extreme momentum exhaustion.
# Volume confirmation ensures institutional participation. Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies).

name = "6h_ElderRay_WilliamsR_VolumeSpike_ATR_v1"
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
    
    # Calculate 1d Williams %R (%R = (Highest High - Close) / (Highest High - Lowest Low) * -100)
    # Williams %R: -100 to 0, readings below -80 = oversold, above -20 = overbought
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r_1d = (highest_high_14 - df_1d['close'].values) / (highest_high_14 - lowest_low_14) * -100
    williams_r_1d[highest_high_14 == lowest_low_14] = -50  # avoid division by zero
    
    # Align Williams %R to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate Bull Power and Bear Power for Elder Ray
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13_1d_aligned
    bear_power = low - ema_13_1d_aligned
    
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
    
    # Start after warmup for EMA, Williams %R, volume, and ATR
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_13_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_atr = atr[i]
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_williams_r = williams_r_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (bullish strength) AND Williams %R < -80 (oversold) AND volume spike
            if curr_bull_power > 0 and curr_williams_r < -80 and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
                lowest_since_entry = curr_close
            # Short: Bear Power < 0 (bearish strength) AND Williams %R > -20 (overbought) AND volume spike
            elif curr_bear_power < 0 and curr_williams_r > -20 and volume_confirm:
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
            
            # Exit conditions: ATR stoploss OR Bull Power <= 0 (loss of bullish strength) OR Williams %R > -20 (overbought)
            stop_price = highest_since_entry - 2.5 * curr_atr
            if curr_close < stop_price or curr_bull_power <= 0 or curr_williams_r > -20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Update lowest low since entry
            if curr_close < lowest_since_entry:
                lowest_since_entry = curr_close
            
            # Exit conditions: ATR stoploss OR Bear Power >= 0 (loss of bearish strength) OR Williams %R < -80 (oversold)
            stop_price = lowest_since_entry + 2.5 * curr_atr
            if curr_close > stop_price or curr_bear_power >= 0 or curr_williams_r < -80:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals