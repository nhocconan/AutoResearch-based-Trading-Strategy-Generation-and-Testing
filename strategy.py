#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w EMA34 trend filter and volume confirmation
# Long: price breaks above Donchian(20) upper band AND price > 1w EMA34 AND volume > 1.5x 20-bar avg
# Short: price breaks below Donchian(20) lower band AND price < 1w EMA34 AND volume > 1.5x 20-bar avg
# Exit: opposite Donchian breakout OR ATR stoploss (2.5 * ATR)
# Donchian channels provide clear breakout levels with built-in volatility adaptation
# 1w EMA34 offers strong trend filter suitable for daily timeframe, reducing whipsaw
# Volume confirmation ensures breakout has institutional participation
# Discrete position sizing: 0.25 for long/short to minimize fee churn
# Target: 40-80 total trades over 4 years (10-20/year) on 1d timeframe

name = "1d_Donchian_Breakout_1wEMA34_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate ATR for stoploss (using 14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr_first = np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])
    tr = np.concatenate([[tr_first], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_upper = rolling_max(high, 20)
    donchian_lower = rolling_min(low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(34, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any indicator is not yet calculated (NaN)
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_ema_1w = ema_34_1w_aligned[i]
        curr_atr = atr[i]
        
        # Volume spike confirmation: current volume > 1.5x 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
        else:
            vol_ma_20 = 0.0
        vol_spike = volume[i] > 1.5 * vol_ma_20 if vol_ma_20 > 0 else False
        
        # Handle exits and stoploss
        if position == 1:  # Long position
            # Stoploss: 2.5 * ATR below entry
            stop_price = entry_price - 2.5 * curr_atr
            # Exit conditions: price breaks below Donchian lower OR stoploss hit
            if curr_low <= donchian_lower[i] or curr_close <= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Stoploss: 2.5 * ATR above entry
            stop_price = entry_price + 2.5 * curr_atr
            # Exit conditions: price breaks above Donchian upper OR stoploss hit
            if curr_high >= donchian_upper[i] or curr_close >= stop_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian upper AND price > 1w EMA34 AND volume spike
            if curr_high > donchian_upper[i] and curr_close > curr_ema_1w and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short entry: price breaks below Donchian lower AND price < 1w EMA34 AND volume spike
            elif curr_low < donchian_lower[i] and curr_close < curr_ema_1w and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals