#!/usr/bin/env python3
# 1h_ema_volume_regime_v1
# Hypothesis: 1h strategy using 50-period EMA for trend direction, volume confirmation (>1.5x 20-period average),
# and chop regime filter (chop < 61.8 = trending) for entries. Uses 4h EMA(50) as HTF trend filter.
# Long when price > 1h EMA(50) AND price > 4h EMA(50) with volume confirmation in trending market.
# Short when price < 1h EMA(50) AND price < 4h EMA(50) with volume confirmation in trending market.
# Exit on opposite EMA cross. Position size = 0.20 to limit drawdown.
# Designed for 1h timeframe: targets 15-30 trades/year (60-120 total over 4 years) by requiring
# confluence of 1h/4h trend, volume, and regime filters to avoid overtrading and fee drag.
# Works in bull/bear markets: EMA captures trend, volume confirms conviction, chop filter avoids whipsaws.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_volume_regime_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1h EMA(50) for trend and signals
    close_s = pd.Series(close)
    ema_50 = close_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Choppiness Index regime filter (14-period)
    atr_period = 14
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    tr_series = pd.Series(tr)
    atr_series = tr_series.rolling(window=atr_period, min_periods=atr_period).mean()
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_high = high_series.rolling(window=atr_period, min_periods=atr_period).max().values
    lowest_low = low_series.rolling(window=atr_period, min_periods=atr_period).min().values
    atr_sum = tr_series.rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(atr_period) / (highest_high - lowest_low))
    
    # Multi-timeframe: 4h EMA(50) trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    close_4h_s = pd.Series(close_4h)
    ema_50_4h = close_4h_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema_50[i]) or np.isnan(ema_50[i-1]) or 
            np.isnan(volume_ma[i]) or np.isnan(chop[i]) or
            np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        # Regime filter: chop < 61.8 indicates trending market
        trending_market = chop[i] < 61.8
        # HTF trend filter: price above/below 4h EMA(50)
        htf_uptrend = close[i] > ema_50_4h_aligned[i]
        htf_downtrend = close[i] < ema_50_4h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 1h EMA(50)
            if close[i] < ema_50[i] and close[i-1] >= ema_50[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1h EMA(50)
            if close[i] > ema_50[i] and close[i-1] <= ema_50[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Check for price/EMA cross with volume, regime, and HTF confirmation
            bullish_cross = (close[i] > ema_50[i] and close[i-1] <= ema_50[i-1]) and volume_confirmed and trending_market and htf_uptrend
            bearish_cross = (close[i] < ema_50[i] and close[i-1] >= ema_50[i-1]) and volume_confirmed and trending_market and htf_downtrend
            
            if bullish_cross:
                position = 1
                signals[i] = 0.20
            elif bearish_cross:
                position = -1
                signals[i] = -0.20
    
    return signals