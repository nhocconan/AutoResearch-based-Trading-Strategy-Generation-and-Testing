#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike_ATRStop_v1
Hypothesis: On 1h timeframe, price breaking Camarilla R1/S1 levels with 4h EMA20 trend alignment and volume confirmation provides robust breakout signals. Uses 1d EMA50 as higher timeframe regime filter to avoid counter-trend trades. Volume confirmation (2.0x average) ensures breakouts have conviction. ATR-based stoploss (2.0x) and discrete sizing (0.0, ±0.20) control risk and minimize fee churn. Targets 60-150 trades over 4 years (15-37/year) to stay within optimal trade frequency for 1h timeframe. Designed to work in both bull (trend following) and bear (mean reversion via regime filter) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Get 1d data for higher timeframe regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate EMA(20) on 4h for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate EMA(50) on 1d for regime filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate ATR(14) for stoploss on 1h
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)  # avoid division by zero
    
    # Calculate Camarilla levels from previous 1h bar
    prev_high = np.concatenate([[np.nan], high[:-1]])
    prev_low = np.concatenate([[np.nan], low[:-1]])
    prev_close = np.concatenate([[np.nan], close[:-1]])
    
    camarilla_r1 = prev_close + ((prev_high - prev_low) * 1.1 / 12)
    camarilla_s1 = prev_close - ((prev_high - prev_low) * 1.1 / 12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 4h EMA(20), 1d EMA(50), ATR(14), volume MA(20)
    start_idx = max(20, 50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ratio[i]) or
            np.isnan(camarilla_r1[i]) or
            np.isnan(camarilla_s1[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        close_val = close[i]
        vol_confirmed = vol_ratio[i] > 2.0  # volume at least 2.0x average (stricter)
        trend_4h_up = close_val > ema_20_4h_aligned[i]
        trend_4h_down = close_val < ema_20_4h_aligned[i]
        regime_up = close_val > ema_50_1d_aligned[i]  # 1d EMA50 as regime filter
        regime_down = close_val < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND 4h trend up AND volume confirmation AND bullish regime
            long_signal = (close_val > camarilla_r1[i]) and trend_4h_up and vol_confirmed and regime_up
            
            # Short: price breaks below Camarilla S1 AND 4h trend down AND volume confirmation AND bearish regime
            short_signal = (close_val < camarilla_s1[i]) and trend_4h_down and vol_confirmed and regime_down
            
            if long_signal:
                signals[i] = 0.20
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.20
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: trend flips down OR regime turns bearish OR price hits ATR stoploss
            if (not trend_4h_up) or (not regime_up) or (close_val < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: trend flips up OR regime turns bullish OR price hits ATR stoploss
            if (not trend_4h_down) or (not regime_down) or (close_val > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike_ATRStop_v1"
timeframe = "1h"
leverage = 1.0