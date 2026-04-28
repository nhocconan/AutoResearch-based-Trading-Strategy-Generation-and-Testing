#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume spike confirmation
# Uses weekly EMA34 to capture primary trend direction. Breaks above Donchian upper in uptrend or below lower in downtrend
# with volume confirmation provide high-probability entries. Target: 7-25 trades/year via tight Donchian breakout
# conditions + volume + trend filter. Works in both bull (breakouts with trend) and bear (mean reversion at extremes).
# Timeframe: 1d, HTF: 1w

name = "1d_Donchian20_1wEMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA34 on 1w close for trend filter
    close_1w = pd.Series(df_1w['close'])
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Donchian channels (20-period) on 1d data
    lookback = 20
    upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align 1w EMA34 to 1d timeframe (completed 1w candles only)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(lookback, 34, 20)  # Donchian20, 1w EMA34, volume MA20 need sufficient history
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        upper_val = upper[i]
        lower_val = lower[i]
        ema34_val = ema34_1w_aligned[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper AND 1w EMA34 uptrend AND volume spike
            if price > upper_val and price > ema34_val and vol_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below lower AND 1w EMA34 downtrend AND volume spike
            elif price < lower_val and price < ema34_val and vol_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price falls below lower (reversal)
            # ATR-based stoploss: 2.0 * ATR below entry (using 1d ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit on stoploss or price < lower (reversal below support)
            if price < stop_loss or price < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or price rises above upper (reversal)
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit on stoploss or price > upper (reversal above resistance)
            if price > stop_loss or price > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals