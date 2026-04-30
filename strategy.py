#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams Fractals with 1d trend filter and volume confirmation
# Williams Fractals identify key swing points; breakouts above/below recent fractals with volume
# and 1d trend alignment capture institutional moves. Designed to work in both bull and bear markets
# by requiring volume confirmation and trend alignment to avoid false breakouts.
# Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe.

name = "12h_WilliamsFractal_Breakout_1dTrend_VolumeSpike_v1"
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
    
    # Load 1d data ONCE before loop for Williams Fractals and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate 1d Williams Fractals
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Bearish fractal: high[n] > high[n-2] and high[n] > high[n-1] and high[n] > high[n+1] and high[n] > high[n+2]
    # Bullish fractal: low[n] < low[n-2] and low[n] < low[n-1] and low[n] < low[n+1] and low[n] < low[n+2]
    bearish_fractal = np.full(len(high_1d), np.nan)
    bullish_fractal = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d) - 2):
        if (high_1d[i] > high_1d[i-2] and high_1d[i] > high_1d[i-1] and 
            high_1d[i] > high_1d[i+1] and high_1d[i] > high_1d[i+2]):
            bearish_fractal[i] = high_1d[i]
        if (low_1d[i] < low_1d[i-2] and low_1d[i] < low_1d[i-1] and 
            low_1d[i] < low_1d[i+1] and low_1d[i] < low_1d[i+2]):
            bullish_fractal[i] = low_1d[i]
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe (wait for completed 1d bar)
    # Williams Fractals need extra 2-bar confirmation delay
    bearish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bearish_fractal, additional_delay_bars=2)
    bullish_fractal_aligned = align_htf_to_ltf(prices, df_1d, bullish_fractal, additional_delay_bars=2)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR(14) for dynamic stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # warmup for EMA(34)
    
    for i in range(start_idx, n):
        # Volume confirmation: volume > 2.0x 30-period average
        vol_ma_30 = np.mean(volume[max(0, i-30):i])
        volume_spike = volume[i] > (2.0 * vol_ma_30)
        
        curr_close = close[i]
        curr_ema = ema_34_1d_aligned[i]
        curr_atr = atr[i]
        
        if position == 0:  # Flat - look for new entries
            # Require volume spike and trend alignment
            if volume_spike:
                # Bullish entry: price breaks above recent bullish fractal with 1d uptrend
                if not np.isnan(bullish_fractal_aligned[i]) and curr_close > bullish_fractal_aligned[i] and curr_close > curr_ema:
                    signals[i] = 0.25
                    position = 1
                    entry_price = curr_close
                # Bearish entry: price breaks below recent bearish fractal with 1d downtrend
                elif not np.isnan(bearish_fractal_aligned[i]) and curr_close < bearish_fractal_aligned[i] and curr_close < curr_ema:
                    signals[i] = -0.25
                    position = -1
                    entry_price = curr_close
        
        elif position == 1:  # Long position
            # Stoploss: 2.0 * ATR below entry price OR price breaks below recent bullish fractal
            if curr_close < entry_price - 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif not np.isnan(bullish_fractal_aligned[i]) and curr_close < bullish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2x ATR above entry
            elif curr_close >= entry_price + 4.0 * curr_atr:
                signals[i] = 0.10  # reduce position
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Stoploss: 2.0 * ATR above entry price OR price breaks above recent bearish fractal
            if curr_close > entry_price + 2.0 * curr_atr:
                signals[i] = 0.0
                position = 0
            elif not np.isnan(bearish_fractal_aligned[i]) and curr_close > bearish_fractal_aligned[i]:
                signals[i] = 0.0
                position = 0
            # Take profit: price reaches 2x ATR below entry
            elif curr_close <= entry_price - 4.0 * curr_atr:
                signals[i] = -0.10  # reduce position
            else:
                signals[i] = -0.25
    
    return signals