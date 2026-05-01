#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 12h ADX regime filter + volume confirmation.
# Long when Bull Power > 0 AND 12h ADX > 25 (trending) AND volume > 1.5x 6h volume average.
# Short when Bear Power < 0 AND 12h ADX > 25 (trending) AND volume > 1.5x 6h volume average.
# Uses Elder Power = Close - EMA13 (Bull) and EMA13 - Close (Bear) for trend strength.
# ADX filter ensures we only trade in trending markets, reducing whipsaw in ranges.
# Volume confirmation adds momentum validity to breakouts.
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years).
# Discrete sizing: 0.25. ATR(14) stoploss: signal→0 when price moves against position by 2.0*ATR.

name = "6h_ElderRay_12hADX_Trend_Volume_v1"
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
    
    # Calculate 12h ADX(14) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # ADX calculation
    plus_dm = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    tr_12h = np.maximum(np.maximum(high_12h[1:] - low_12h[1:], 
                                   np.abs(high_12h[1:] - close_12h[:-1])), 
                        np.abs(low_12h[1:] - close_12h[:-1]))
    tr_12h = np.concatenate([[np.max([high_12h[0] - low_12h[0], 
                                     np.abs(high_12h[0] - close_12h[0]), 
                                     np.abs(low_12h[0] - close_12h[0])])], tr_12h])
    
    atr_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean().values / atr_12h
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = Close - EMA13, Bear Power = EMA13 - Close
    bull_power = close - ema_13
    bear_power = ema_13 - close
    
    # Calculate 6h volume average (20-period)
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for ATR, EMA, ADX, and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(atr[i]) or 
            np.isnan(adx_12h_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma_6h[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 6h volume average
        if vol_ma_6h[i] <= 0 or np.isnan(vol_ma_6h[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_ma_6h[i] * 1.5)
        
        # Trend filter: 12h ADX > 25 indicates trending market
        trending = adx_12h_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 AND trending AND volume confirmation
            if (bull_power[i] > 0 and 
                trending and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: Bear Power > 0 AND trending AND volume confirmation
            elif (bear_power[i] > 0 and 
                  trending and 
                  volume_confirm):
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
            # Exit: Bull Power turns negative OR ADX drops below 20 (range)
            elif (bull_power[i] <= 0) or (adx_12h_aligned[i] < 20):
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
            # Exit: Bear Power turns negative OR ADX drops below 20 (range)
            elif (bear_power[i] <= 0) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals