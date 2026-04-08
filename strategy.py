# [EXPERIMENT #22783] 4h_1d_donchian_breakout_volume_v3
# Hypothesis: Breakout of 4h Donchian channels with 1d EMA trend filter and volume confirmation works in both bull and bear markets by capturing momentum bursts while avoiding countertrend trades. 4h timeframe limits overtrading; volume and trend filters reduce false breakouts. This version adds a choppiness regime filter to avoid whipsaws in sideways markets, targeting 20-50 trades/year per symbol.
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 4h high/low
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    # Align to 4h timeframe (no additional delay needed for Donchian)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Calculate EMA50 on 1d close
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Align to 4h timeframe (no additional delay needed for EMA)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 4h volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    # Choppiness regime filter (1d): avoid entries in choppy markets
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    atr_1d = np.zeros(len(close_1d_arr))
    for i in range(1, len(close_1d_arr)):
        tr = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d_arr[i-1]), abs(low_1d[i] - close_1d_arr[i-1]))
        atr_1d[i] = 0.99 * atr_1d[i-1] + 0.01 * tr if i > 1 else tr
    # Choppiness = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
    chop_period = 14
    sum_atr = np.zeros(len(close_1d_arr))
    max_high = np.zeros(len(close_1d_arr))
    min_low = np.zeros(len(close_1d_arr))
    for i in range(chop_period, len(close_1d_arr)):
        sum_atr[i] = np.sum(atr_1d[i-chop_period+1:i+1])
        max_high[i] = np.max(high_1d[i-chop_period+1:i+1])
        min_low[i] = np.min(low_1d[i-chop_period+1:i+1])
    chop = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(chop_period)
    chop = np.where((max_high - min_low) > 0, chop, 50)  # avoid div by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Only trade in trending markets (Chop < 61.8) or strong breakouts
        is_trending = chop_aligned[i] < 61.8
        
        if position == 1:  # Long position
            # Exit: price closes below 4h Donchian low
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above 4h Donchian high
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price breaks above 4h Donchian high, above 1d EMA50, with volume confirmation, in trending market
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_1d_aligned[i] and vol_confirm[i] and is_trending:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 4h Donchian low, below 1d EMA50, with volume confirmation, in trending market
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_1d_aligned[i] and vol_confirm[i] and is_trending:
                position = -1
                signals[i] = -0.25
    
    return signals