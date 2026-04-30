#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume spike
# Uses discrete sizing 0.25 to minimize fee churn. Target: 75-200 trades over 4 years (19-50/year).
# Works in bull markets (breakouts continue trend) and bear markets (breakdowns continue downtrend).
# Added ATR-based stoploss to control drawdown.
# Focus on BTC/ETH as primary symbols; SOL as secondary.

name = "4h_Donchian20_12hEMA34_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter (MTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Donchian channels (20-period) on primary timeframe
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period average (stricter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # ATR for stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(100, 34, 20, 20, 14)  # warmup
    
    for i in range(start_idx, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(highest_20[i]) or
            np.isnan(lowest_20[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_34_12h = ema_34_12h_aligned[i]
        curr_highest_20 = highest_20[i]
        curr_lowest_20 = lowest_20[i]
        curr_volume_spike = volume_spike[i]
        curr_atr = atr_14[i]
        
        if position == 0:  # Flat - look for new entries
            # Only trade on volume spike with Donchian break and 12h trend filter
            if curr_volume_spike:
                # Bullish: Close breaks above upper Donchian + price above 12h EMA34
                if curr_close > curr_highest_20 and curr_close > curr_ema_34_12h:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish: Close breaks below lower Donchian + price below 12h EMA34
                elif curr_close < curr_lowest_20 and curr_close < curr_ema_34_12h:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry (wider for 4h)
            stop_loss = entry_price - 2.5 * curr_atr
            # Exit: Stoploss hit OR close drops below upper Donchian OR loses 12h trend
            if curr_low <= stop_loss or curr_close < curr_highest_20 or curr_close < curr_ema_34_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_loss = entry_price + 2.5 * curr_atr
            # Exit: Stoploss hit OR close rises above lower Donchian OR loses 12h trend
            if curr_high >= stop_loss or curr_close > curr_lowest_20 or curr_close > curr_ema_34_12h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals