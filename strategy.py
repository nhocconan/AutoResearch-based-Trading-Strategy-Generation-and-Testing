#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation.
# Long when price breaks above 20-period Donchian high AND price > 1d EMA34 AND volume > 1.5x 20-bar average.
# Short when price breaks below 20-period Donchian low AND price < 1d EMA34 AND volume > 1.5x 20-bar average.
# Exit when price crosses Donchian midpoint OR ATR-based stoploss (2.0x ATR).
# Uses discrete position sizing (0.25) to limit drawdown and fee churn.
# Donchian provides clear structure; EMA34 filters major trend; volume confirms breakout strength.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull/bear via 1d EMA34 trend filter.

name = "12h_Donchian20_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: 20-period rolling maximum of high
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: 20-period rolling minimum of low
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint: average of high and low channels
    donchian_mid = (donchian_high_20 + donchian_low_20) / 2
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # ATR for stoploss (using 12h data)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20, 14)  # warmup for EMA34, Donchian, ATR
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or 
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
            # Long: break above Donchian high, uptrend (price > 1d EMA34), volume confirmation
            if (curr_high > donchian_high_aligned[i] and 
                curr_close > ema_34_1d_aligned[i] and 
                curr_volume_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: break below Donchian low, downtrend (price < 1d EMA34), volume confirmation
            elif (curr_low < donchian_low_aligned[i] and 
                  curr_close < ema_34_1d_aligned[i] and 
                  curr_volume_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        
        elif position == 1:  # Long position
            # Exit conditions: Donchian midpoint cross OR ATR stoploss
            exit_signal = False
            if curr_close < donchian_mid_aligned[i]:  # midpoint cross
                exit_signal = True
            elif curr_close < entry_price - 2.0 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions: Donchian midpoint cross OR ATR stoploss
            exit_signal = False
            if curr_close > donchian_mid_aligned[i]:  # midpoint cross
                exit_signal = True
            elif curr_close > entry_price + 2.0 * atr[i]:  # ATR stoploss
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals