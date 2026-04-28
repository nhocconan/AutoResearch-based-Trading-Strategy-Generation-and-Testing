#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA(34) trend filter
# Long when price breaks above 20-bar Donchian high + volume > 1.5x 20-bar average + price above 1d EMA(34)
# Short when price breaks below 20-bar Donchian low + volume > 1.5x 20-bar average + price below 1d EMA(34)
# Uses ATR-based trailing stop (3.0 * ATR from extreme) to manage risk and minimize whipsaw
# Targets 20-50 trades/year on 4h timeframe to avoid fee drag while capturing medium-term trends
# Works in bull markets via breakouts with trend alignment and in bear markets via breakdowns with trend filter

name = "4h_Donchian20_Breakout_VolumeSpike_1dEMA34_Trend_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
        donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    # ATR calculation for dynamic stoploss (14-period)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(np.maximum(tr1, tr2), tr3)])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = max(lookback - 1, 20, 34, 14)  # Donchian(20), volume MA(20), 1d EMA(34), ATR(14)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_confirm = volume_spike[i]
        price_above_ema = price > ema_34_1d_aligned[i]
        price_below_ema = price < ema_34_1d_aligned[i]
        
        # Handle position exits and stops
        if position == 1:  # Long position
            # Update trailing stop: highest high since entry minus 3.0 * ATR
            long_stop = max(long_stop, high[i] - 3.0 * atr[i])
            if price < long_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Update trailing stop: lowest low since entry plus 3.0 * ATR
            short_stop = min(short_stop, low[i] + 3.0 * atr[i])
            if price > short_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high + volume spike + price above 1d EMA
            if price > donchian_high[i] and vol_confirm and price_above_ema:
                signals[i] = 0.25
                position = 1
                entry_price = price
                long_stop = high[i] - 3.0 * atr[i]  # Initial stop
            # Short entry: price breaks below Donchian low + volume spike + price below 1d EMA
            elif price < donchian_low[i] and vol_confirm and price_below_ema:
                signals[i] = -0.25
                position = -1
                entry_price = price
                short_stop = low[i] + 3.0 * atr[i]  # Initial stop
            else:
                signals[i] = 0.0
    
    return signals