#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get weekly data for EMA(50) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily True Range and ATR(14)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily High/Low/Close for Choppiness Index
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Daily True Range for chop calculation
    tr_daily = np.maximum(daily_high - daily_low, np.maximum(np.abs(daily_high - np.roll(daily_close, 1)), np.abs(daily_low - np.roll(daily_close, 1))))
    tr_daily[0] = daily_high[0] - daily_low[0]
    atr14_daily = pd.Series(tr_daily).rolling(window=14, min_periods=14).sum().values
    
    # Daily max/min over 14 periods
    max_high14 = pd.Series(daily_high).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(daily_low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(TR14)/(max(HH14)-min(LL14))) / log10(14)
    range14 = max_high14 - min_low14
    chop = 100 * np.log10(atr14_daily / range14) / np.log10(14)
    chop = np.where(range14 > 0, chop, 50)  # Avoid division by zero
    
    # Weekly EMA(50) for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 12h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need chop, weekly EMA50, and volume data
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(chop_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr14[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        ema_trend = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        atr_val = atr14[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        vol_filter = vol_current > (vol_ma_val * 1.5)
        
        # Chop regime: > 61.8 = range (mean revert), < 38.2 = trending (trend follow)
        is_range = chop_val > 61.8
        is_trend = chop_val < 38.2
        
        if position == 0:
            # In range: mean reversion at Bollinger-like bands (using ATR)
            if is_range:
                # Calculate dynamic bands: ±1.5 * ATR from EMA20
                ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                upper_band = ema20[i] + 1.5 * atr_val
                lower_band = ema20[i] - 1.5 * atr_val
                
                # Long at lower band with volume
                if close[i] <= lower_band and vol_filter:
                    signals[i] = size
                    position = 1
                # Short at upper band with volume
                elif close[i] >= upper_band and vol_filter:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            # In trend: follow trend with pullbacks
            elif is_trend:
                # Long: pullback to EMA20 in uptrend
                ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
                if close[i] <= ema20[i] and close[i] > ema_trend and vol_filter:
                    signals[i] = size
                    position = 1
                # Short: pullback to EMA20 in downtrend
                elif close[i] >= ema20[i] and close[i] < ema_trend and vol_filter:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            # Neutral chop: no trade
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: chop shifts to trend and price breaks above EMA20, or trend reversal
            ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
            if (is_trend and close[i] >= ema20[i]) or (close[i] < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: chop shifts to trend and price breaks below EMA20, or trend reversal
            ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
            if (is_trend and close[i] <= ema20[i]) or (close[i] > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_ChopRegime_EMA20_Pullback_VolumeFilter"
timeframe = "12h"
leverage = 1.0