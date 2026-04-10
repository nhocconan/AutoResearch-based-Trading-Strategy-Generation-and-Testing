#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d EMA trend filter and volume confirmation
# - Long when price breaks above Donchian(20) high AND 1d close > EMA50 AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 1d close < EMA50 AND volume > 1.5x 20-bar avg
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.5x ATR(14)
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - Donchian breakouts capture strong momentum moves; 1d EMA filter ensures alignment with higher timeframe trend
# - Volume confirmation reduces false breakouts
# - Works in both bull and bear markets by trading breakouts in direction of 1d trend

name = "4h_1d_donchian_breakout_volume_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.5 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Pre-compute Donchian channels on 4h data
    donchian_window = 20
    highest_high = prices['high'].rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = prices['low'].rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Pre-compute ATR(14) for stoploss
    high_low = prices['high'] - prices['low']
    high_close = np.abs(prices['high'] - prices['close'].shift(1))
    low_close = np.abs(prices['low'] - prices['close'].shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(true_range).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: price breaks above Donchian high in 1d uptrend with volume spike
            if (prices['close'].iloc[i] > highest_high[i] and 
                prices['close'].iloc[i] > ema_50_1d_aligned[i] and 
                vol_spike_1d_aligned[i]):
                position = 1
                entry_price = prices['close'].iloc[i]
                signals[i] = 0.25
            # Short signal: price breaks below Donchian low in 1d downtrend with volume spike
            elif (prices['close'].iloc[i] < lowest_low[i] and 
                  prices['close'].iloc[i] < ema_50_1d_aligned[i] and 
                  vol_spike_1d_aligned[i]):
                position = -1
                entry_price = prices['close'].iloc[i]
                signals[i] = -0.25
        else:  # Have position - look for exit
            # ATR-based stoploss
            if position == 1:  # Long position
                stop_price = entry_price - 2.5 * atr[i]
                # Exit if price closes below stop or Donchian low breaks
                if (prices['close'].iloc[i] < stop_price or 
                    prices['close'].iloc[i] < lowest_low[i]):
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # Short position
                stop_price = entry_price + 2.5 * atr[i]
                # Exit if price closes above stop or Donchian high breaks
                if (prices['close'].iloc[i] > stop_price or 
                    prices['close'].iloc[i] > highest_high[i]):
                    position = 0
                    entry_price = 0.0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals