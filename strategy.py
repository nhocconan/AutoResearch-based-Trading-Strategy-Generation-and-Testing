#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper band AND price > 1w EMA50 AND volume > 1.5x 20-bar avg
# Short when price breaks below Donchian lower band AND price < 1w EMA50 AND volume > 1.5x 20-bar avg
# Exit when price crosses opposite Donchian band or trailing ATR stop (2.5x ATR(20))
# Uses discrete position sizing (0.25) to minimize fee churn while capturing multi-day trends.
# Target: 30-100 total trades over 4 years (7-25/year) on 1d.
# Donchian(20) captures multi-day breakouts, 1w EMA50 filters counter-trend moves on weekly trend,
# volume confirmation ensures institutional participation. Works in bull markets (trend continuation) 
# and bear markets (mean reversion within trend via exits).

name = "1d_Donchian20_1wEMA50_VolumeConfirm_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w data
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 1d timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR(20) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period) using prior bar's data to avoid look-ahead
    # We need the highest high and lowest low of the prior 20 completed bars
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for ATR stop
    
    start_idx = max(50, 20) + 1  # EMA50 warmup + Donchian warmup + 1 for shift
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_50 = ema_50_1w_aligned[i]
        upper_band = highest_high[i]
        lower_band = lowest_low[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit conditions: price crosses below lower band OR ATR stoploss hit
            if curr_close < lower_band or curr_low < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price crosses above upper band OR ATR stoploss hit
            if curr_close > upper_band or curr_high > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price breaks above upper band AND price > 1w EMA50 AND volume confirmation
            if curr_close > upper_band and curr_close > ema_50 and vol_conf:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short when price breaks below lower band AND price < 1w EMA50 AND volume confirmation
            elif curr_close < lower_band and curr_close < ema_50 and vol_conf:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
    
    return signals