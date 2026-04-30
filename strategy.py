#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation + ATR stoploss.
# Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.8x 20-bar average.
# Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.8x 20-bar average.
# Exit when price crosses Donchian(10) midline OR ATR-based stoploss (2x ATR).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Works in bull/bear via 1d EMA50 trend filter. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_1dEMA50_Trend_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channels
    donchian_20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_10_high = pd.Series(high).rolling(window=10, min_periods=10).max().values
    donchian_10_low = pd.Series(low).rolling(window=10, min_periods=10).min().values
    donchian_10_mid = (donchian_10_high + donchian_10_low) / 2
    
    # ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(50, 20, 14)  # warmup for EMA50, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_20_high[i]) or 
            np.isnan(donchian_20_low[i]) or np.isnan(donchian_10_mid[i]) or 
            np.isnan(atr[i]) or np.isnan(volume_confirm[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: break above Donchian(20) high, uptrend, volume confirmation
            if (curr_high > donchian_20_high[i] and 
                curr_close > ema_50_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Donchian(20) low, downtrend, volume confirmation
            elif (curr_low < donchian_20_low[i] and 
                  curr_close < ema_50_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Donchian(10) midline cross OR ATR stoploss
            exit_signal = False
            if curr_close < donchian_10_mid[i]:  # midline cross
                exit_signal = True
            elif curr_close < entry_price - 2.0 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Donchian(10) midline cross OR ATR stoploss
            exit_signal = False
            if curr_close > donchian_10_mid[i]:  # midline cross
                exit_signal = True
            elif curr_close > entry_price + 2.0 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals