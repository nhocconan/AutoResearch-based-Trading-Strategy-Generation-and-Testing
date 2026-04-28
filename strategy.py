#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Fractal Breakout with 12h trend filter and volume confirmation
# Williams Fractals identify potential reversal points: bearish fractal = high with 2 lower highs on each side,
# bullish fractal = low with 2 higher lows on each side. Long when price breaks above bearish fractal
# with 12h uptrend (price > 12h EMA50) and volume confirmation. Short when price breaks below bullish
# fractal with 12h downtrend (price < 12h EMA50) and volume confirmation. Uses 6h timeframe targeting
# 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag. Works in trending markets via
# breakout confirmation and in ranging markets via fade at extreme fractals (though primary logic is breakout).

name = "6h_WilliamsFractal_Breakout_12hEMA50_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams Fractals on 6h data
    # Bearish fractal: high[i] > high[i-1] and high[i] > high[i-2] and high[i] > high[i+1] and high[i] > high[i+2]
    # Bullish fractal: low[i] < low[i-1] and low[i] < low[i-2] and low[i] < low[i+1] and low[i] < low[i+2]
    bearish_fractal = np.full(n, np.nan)
    bullish_fractal = np.full(n, np.nan)
    
    for i in range(2, n-2):
        if (high[i] > high[i-1] and high[i] > high[i-2] and 
            high[i] > high[i+1] and high[i] > high[i+2]):
            bearish_fractal[i] = high[i]
        if (low[i] < low[i-1] and low[i] < low[i-2] and 
            low[i] < low[i+1] and low[i] < low[i+2]):
            bullish_fractal[i] = low[i]
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = max(20, 50)  # volume MA(20), 12h EMA(50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_ma_20[i]) or 
            np.isnan(bearish_fractal[i]) or np.isnan(bullish_fractal[i])):
            signals[i] = 0.0
            continue
        
        vol_confirm = volume_spike[i]
        price = close[i]
        curr_bear_fractal = bearish_fractal[i]
        curr_bull_fractal = bullish_fractal[i]
        prev_bear_fractal = bearish_fractal[i-1] if i > 0 else np.nan
        prev_bull_fractal = bullish_fractal[i-1] if i > 0 else np.nan
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above bearish fractal, 12h uptrend, volume spike
            if (not np.isnan(curr_bear_fractal) and price > curr_bear_fractal and 
                price > ema_50_12h_aligned[i] and vol_confirm):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below bullish fractal, 12h downtrend, volume spike
            elif (not np.isnan(curr_bull_fractal) and price < curr_bull_fractal and 
                  price < ema_50_12h_aligned[i] and vol_confirm):
                signals[i] = -0.25
                position = -1
                entry_price = price
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on stoploss or price re-enters fractal level
            # ATR-based stoploss: 2.0 * ATR below entry (using 6h ATR)
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price - 2.0 * atr_val
            # Exit if stoploss hit or price re-enters below bearish fractal (failed breakout)
            if price < stop_loss or (not np.isnan(curr_bear_fractal) and price < curr_bear_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short - exit on stoploss or price re-enters fractal level
            # ATR-based stoploss: 2.0 * ATR above entry
            tr1 = high[max(0, i-1):i+1] - low[max(0, i-1):i+1]
            tr2 = np.abs(high[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr3 = np.abs(low[max(0, i-1):i+1] - close[max(0, i-1):i])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            atr_val = np.mean(tr[-14:]) if len(tr) >= 14 else np.mean(tr)
            stop_loss = entry_price + 2.0 * atr_val
            # Exit if stoploss hit or price re-enters above bullish fractal (failed breakout)
            if price > stop_loss or (not np.isnan(curr_bull_fractal) and price > curr_bull_fractal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals