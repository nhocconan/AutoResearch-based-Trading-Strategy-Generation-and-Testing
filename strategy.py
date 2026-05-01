#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme with 1d ADX trend filter and volume confirmation.
# Long when Williams %R < -80 (oversold) AND 1d ADX > 25 (trending) AND volume > 1.5x 20-period median.
# Short when Williams %R > -20 (overbought) AND 1d ADX > 25 AND volume > 1.5x 20-period median.
# Williams %R identifies exhaustion points in ranging markets; ADX filter ensures we only trade during trending conditions to avoid false signals in chop.
# Volume confirmation adds conviction to the reversal signal.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).
# Target: 12-37 trades/year on 6h timeframe (~50-150 total over 4 years).

name = "6h_WilliamsR_Extreme_1dADX_Volume_v2"
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
    
    # Calculate ATR(14) for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams %R(14) - using prior bar to avoid look-ahead
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().shift(1).values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().shift(1).values
    williams_r = np.where((highest_high_14 - lowest_low_14) != 0,
                          -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14),
                          0)
    
    # Calculate 1d ADX(14) trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX components
    plus_dm = np.where((df_1d['high'].values[1:] - df_1d['high'].values[:-1]) > 
                       (df_1d['low'].values[:-1] - df_1d['low'].values[1:]),
                       np.maximum(df_1d['high'].values[1:] - df_1d['high'].values[:-1], 0), 0)
    minus_dm = np.where((df_1d['low'].values[:-1] - df_1d['low'].values[1:]) > 
                        (df_1d['high'].values[1:] - df_1d['high'].values[:-1]),
                        np.maximum(df_1d['low'].values[:-1] - df_1d['low'].values[1:], 0), 0)
    
    # Handle first element
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # True Range for ADX
    tr_1d = np.maximum(np.maximum(df_1d['high'].values - df_1d['low'].values,
                                  np.abs(df_1d['high'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]])),
                                  np.abs(df_1d['low'].values - np.concatenate([[df_1d['close'].values[0]], df_1d['close'].values[:-1]]))))
    
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    plus_di_1d = np.where(atr_1d != 0, 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr_1d, 0)
    minus_di_1d = np.where(atr_1d != 0, 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr_1d, 0)
    
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0,
                     100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d),
                     0)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, Williams %R, volume, and ADX
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_median_20[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        curr_williams_r = williams_r[i]
        
        # Trend filter: 1d ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: Williams %R < -80 (oversold) AND trending AND volume spike
            if curr_williams_r < -80 and trending and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Williams %R > -20 (overbought) AND trending AND volume spike
            elif curr_williams_r > -20 and trending and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R > -50 (exiting oversold) OR ADX < 20 (trend weakening)
            elif curr_williams_r > -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: price moves against position by 2.0*ATR
            if curr_close > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: Williams %R < -50 (exiting overbought) OR ADX < 20 (trend weakening)
            elif curr_williams_r < -50 or adx_1d_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals