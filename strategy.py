#!/usr/bin/env python3
"""
4h_Three_Inside_Up_Down_Trend_Confirm
Hypothesis: Uses Three Inside Up/Down candlestick patterns for reversal signals, confirmed by 1-day EMA trend and volume spike.
Designed for low trade frequency (<25/year) to avoid fee drag while capturing high-probability reversals in both bull and bear markets.
"""

name = "4h_Three_Inside_Up_Down_Trend_Confirm"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1d EMA34 for trend filter ---
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # --- Volume Spike Detection (2x 20-period EMA) ---
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ema.values)
    
    # --- Three Inside Up/Down Pattern Detection ---
    # Three Inside Up: Bearish candle, then bullish engulfing, then bullish confirmation
    # Three Inside Down: Bullish candle, then bearish engulfing, then bearish confirmation
    
    # Bullish engulfing: current candle engulfs previous bearish candle
    bullish_engulfing = (close > open_) & (open_ < close) & (close > open_[1]) & (open_ < close[1]) & (close[1] < open_[1])
    # Bearish engulfing: current candle engulfs previous bullish candle
    bearish_engulfing = (close < open_) & (open_ > close) & (close < open_[1]) & (open_ > close[1]) & (close[1] > open_[1])
    
    # Three Inside Up: previous candle was bearish, then bullish engulfing, then current bullish
    three_inside_up = (close[1] < open_[1]) & bullish_engulfing & (close > open_)
    # Three Inside Down: previous candle was bullish, then bearish engulfing, then current bearish
    three_inside_down = (close[1] > open_[1]) & bearish_engulfing & (close < open_)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_spike[i])):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend based on price vs EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            if price_above_ema:
                # Uptrend: look for Three Inside Down (bearish reversal) for short
                if three_inside_down[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
            elif price_below_ema:
                # Downtrend: look for Three Inside Up (bullish reversal) for long
                if three_inside_up[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
        else:
            # Exit conditions: reverse signal or trend failure
            if position == 1:
                # Exit long: Three Inside Down forms or price breaks below EMA
                if three_inside_down[i] or (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: Three Inside Up forms or price breaks above EMA
                if three_inside_up[i] or (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals