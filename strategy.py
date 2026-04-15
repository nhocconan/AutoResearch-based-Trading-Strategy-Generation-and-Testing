#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and volume confirmation
# Long when 1h price < Bollinger lower band (20,2) + 4h EMA50 > EMA200 (bullish trend) + volume > 1.5x 20-period avg
# Short when 1h price > Bollinger upper band (20,2) + 4h EMA50 < EMA200 (bearish trend) + volume > 1.5x 20-period avg
# Uses Bollinger bands for mean reversion entries in ranging markets, filtered by 4h trend direction to avoid counter-trend trades.
# Volume confirmation reduces false breakouts. Designed for low trade frequency (15-35/year) on 1h timeframe.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: EMA50 and EMA200 (trend filter) ===
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema200_4h)
    
    # === 1h Indicators: Bollinger Bands (20,2) ===
    bb_window = 20
    bb_std = 2
    sma_20 = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    std_20 = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    bb_upper = sma_20 + (bb_std * std_20)
    bb_lower = sma_20 - (bb_std * std_20)
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(bb_window, 200) + 20  # Bollinger(20) + EMA200(200) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or
            np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_4h_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine 4h trend: bullish if EMA50 > EMA200, bearish if EMA50 < EMA200
        bullish_trend = ema50_4h_aligned[i] > ema200_4h_aligned[i]
        bearish_trend = ema50_4h_aligned[i] < ema200_4h_aligned[i]
        
        # === LONG CONDITIONS ===
        # 1. Price touches/below Bollinger lower band (mean reversion long)
        # 2. 4h trend is bullish (only long in uptrend)
        # 3. Volume confirmation
        if (close[i] <= bb_lower[i]) and \
           bullish_trend and vol_confirm:
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price touches/above Bollinger upper band (mean reversion short)
        # 2. 4h trend is bearish (only short in downtrend)
        # 3. Volume confirmation
        elif (close[i] >= bb_upper[i]) and \
             bearish_trend and vol_confirm:
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_BB20_2_4hEMA50_200_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0