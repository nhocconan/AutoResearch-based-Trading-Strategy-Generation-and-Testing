#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly trend filter, volume confirmation, and ATR-based exit
# Works in bull markets via breakouts, in bear via short breakdowns
# Weekly trend prevents counter-trend trades, reducing whipsaw
# Volume confirmation ensures breakout authenticity
# Target: 8-15 trades/year, low frequency to minimize fee drag

name = "1d_Donchian20_WeeklyTrend_Volume_ATRExit"
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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA34 for trend determination
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need both Donchian and EMA warmup
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        weekly_ema = ema_34_1w_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: break above 20-day high with volume and weekly uptrend
            if price > high_20[i] and volume_confirmed and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: break below 20-day low with volume and weekly downtrend
            elif price < low_20[i] and volume_confirmed and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below 20-day low or ATR-based stop
            atr_period = 14
            if i >= atr_period:
                tr = np.maximum(high[i-atr_period+1:i+1] - low[i-atr_period+1:i+1],
                               np.absolute(high[i-atr_period+1:i+1] - close[i-atr_period:i]))
                tr = np.maximum(tr, np.absolute(low[i-atr_period+1:i+1] - close[i-atr_period:i]))
                atr = np.mean(tr)
                if price < low_20[i] or price < close[i-1] - 2.5 * atr:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above 20-day high or ATR-based stop
            atr_period = 14
            if i >= atr_period:
                tr = np.maximum(high[i-atr_period+1:i+1] - low[i-atr_period+1:i+1],
                               np.absolute(high[i-atr_period+1:i+1] - close[i-atr_period:i]))
                tr = np.maximum(tr, np.absolute(low[i-atr_period+1:i+1] - close[i-atr_period:i]))
                atr = np.mean(tr)
                if price > high_20[i] or price > close[i-1] + 2.5 * atr:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25
    
    return signals