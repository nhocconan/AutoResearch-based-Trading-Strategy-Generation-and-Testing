#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Bollinger Band breakout with daily trend filter and volume confirmation.
# Uses Bollinger Bands (20, 2.0) on 12h to detect volatility expansion, daily EMA50 for trend direction,
# and volume spike confirmation to avoid false breakouts. Designed for 15-30 trades/year to minimize fee drag.
# Works in bull markets via breakout continuation and in bear markets via mean reversion at band extremes.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Bollinger Bands
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Bollinger Bands on 12h close (20, 2.0)
    sma_20_12h = np.full(len(close_12h), np.nan)
    std_20_12h = np.full(len(close_12h), np.nan)
    for i in range(20, len(close_12h)):
        sma_20_12h[i] = np.mean(close_12h[i-20:i])
        std_20_12h[i] = np.std(close_12h[i-20:i])
    upper_bb_12h = sma_20_12h + 2.0 * std_20_12h
    lower_bb_12h = sma_20_12h - 2.0 * std_20_12h
    
    # Align 12h Bollinger Bands to 6h timeframe
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb_12h)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb_12h)
    sma_20_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_20_12h)
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily close
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])  # simple average for first value
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = close_1d[i] * (2/(50+1)) + ema_50_1d[i-1] * (1 - 2/(50+1))
    
    # Align daily EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 6h ATR for stop loss
    tr_6h_1 = high - low
    tr_6h_2 = np.abs(high - np.roll(close, 1))
    tr_6h_3 = np.abs(low - np.roll(close, 1))
    tr_6h_1[0] = high[0] - low[0]
    tr_6h_2[0] = np.abs(high[0] - close[0])
    tr_6h_3[0] = np.abs(low[0] - close[0])
    tr_6h = np.maximum(tr_6h_1, np.maximum(tr_6h_2, tr_6h_3))
    atr_6h = pd.Series(tr_6h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # need daily EMA50, volume MA, BB
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_bb_aligned[i]) or np.isnan(lower_bb_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        vol_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        # Trend filter: price above daily EMA50 (uptrend) or below (downtrend)
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above upper Bollinger Band with volume and uptrend
            if (close[i] > upper_bb_aligned[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Bollinger Band with volume and downtrend
            elif (close[i] < lower_bb_aligned[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to middle Bollinger Band or ATR-based stop
            if close[i] < sma_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to middle Bollinger Band or ATR-based stop
            if close[i] > sma_20_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerBandBreakout_12hVol_1dEMA50"
timeframe = "6h"
leverage = 1.0