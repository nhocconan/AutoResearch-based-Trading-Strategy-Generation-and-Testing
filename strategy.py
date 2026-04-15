#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Band squeeze breakout with volume confirmation and 1d trend filter
# Long when price breaks above upper BB(20,2) + BB width < 20th percentile (squeeze) + volume > 1.5x avg + 1d close > 1d EMA50
# Short when price breaks below lower BB(20,2) + BB width < 20th percentile + volume > 1.5x avg + 1d close < 1d EMA50
# Designed for low trade frequency (20-40/year) to minimize fee drag while capturing volatility expansion in ranging markets

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Bollinger Bands (20,2) ===
    close_4h = df_4h['close'].values
    sma_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_4h).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)
    bb_width = (upper_bb - lower_bb) / sma_20  # normalized width
    
    upper_bb_aligned = align_htf_to_ltf(prices, df_4h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_4h, lower_bb)
    bb_width_aligned = align_htf_to_ltf(prices, df_4h, bb_width)
    
    # === 1d Indicators: EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # === 1d Indicators: BB width percentile for squeeze detection ===
    # Calculate 1d BB width for regime filter
    sma_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20_1d = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb_1d = sma_20_1d + (2 * std_20_1d)
    lower_bb_1d = sma_20_1d - (2 * std_20_1d)
    bb_width_1d = (upper_bb_1d - lower_bb_1d) / sma_20_1d
    
    # 20th percentile of BB width over 50 periods
    bb_width_percentile = pd.Series(bb_width_1d).rolling(window=50, min_periods=50).quantile(0.20).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or
            np.isnan(bb_width_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h upper Bollinger Band
        # 2. Bollinger Band squeeze (width < 20th percentile)
        # 3. Volume confirmation
        # 4. 1d uptrend (close > EMA50)
        if (close[i] > upper_bb_aligned[i]) and \
           (bb_width_aligned[i] < bb_width_percentile_aligned[i]) and \
           vol_confirm and \
           (close_1d[-1] > ema_50[-1] if len(close_1d) > 0 else False):  # Use last available 1d close
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h lower Bollinger Band
        # 2. Bollinger Band squeeze (width < 20th percentile)
        # 3. Volume confirmation
        # 4. 1d downtrend (close < EMA50)
        elif (close[i] < lower_bb_aligned[i]) and \
             (bb_width_aligned[i] < bb_width_percentile_aligned[i]) and \
             vol_confirm and \
             (close_1d[-1] < ema_50[-1] if len(close_1d) > 0 else False):  # Use last available 1d close
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_BB20_Squeeze_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0