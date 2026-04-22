# 1d_TRIX_Volume_Spike_Momentum
# Hypothesis: TRIX (15) momentum combined with volume spikes and volatility regime filter
# captures trend acceleration phases. Works in both bull and bear markets by catching
# sharp moves after consolidation. Uses 1d timeframe with 1w trend filter to avoid
# counter-trend trades. Low trade frequency (<25/year) minimizes fee drag.
# TRIX > 0 indicates bullish momentum, < 0 bearish. Volume spike confirms conviction.
# Bollinger Band width < 50th percentile identifies low-volatility consolidation
# preceding explosive moves.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for TRIX and Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate TRIX (15) - triple smoothed ROC
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15)
    ema1 = pd.Series(close_1d).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    # ROC of triple EMA
    trix = np.full_like(close_1d, np.nan)
    trix[15:] = (ema3[15:] - ema3[14:-1]) / ema3[14:-1] * 100
    
    # Bollinger Bands (20, 2) for volatility regime
    sma20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    bb_width = bb_upper - bb_lower
    # Percentile of BB width (252-day lookback ~ 1 year)
    bb_width_pct = np.full_like(bb_width, np.nan)
    for i in range(252, len(bb_width)):
        bb_width_pct[i] = pd.Series(bb_width[i-252:i]).rank(pct=True).iloc[-1] * 100
    
    # 1w EMA50 for trend filter (only trade with higher timeframe trend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Align 1d indicators to lower timeframe (1d)
    trix_aligned = align_htf_to_ltf(prices, df_1d, trix)
    bb_width_pct_aligned = align_htf_to_ltf(prices, df_1d, bb_width_pct)
    sma20_aligned = align_htf_to_ltf(prices, df_1d, sma20)
    
    # Volume spike filter (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any data is not ready
        if (np.isnan(trix_aligned[i]) or 
            np.isnan(bb_width_pct_aligned[i]) or 
            np.isnan(sma20_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        trix_val = trix_aligned[i]
        bb_width_pct_val = bb_width_pct_aligned[i]
        sma20_val = sma20_aligned[i]
        ema50_1w = ema50_1w_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-day average (strong conviction)
        vol_spike = vol > 2.0 * vol_ma
        
        # Volatility regime: low volatility (BB width < 50th percentile) precedes breakout
        low_vol = bb_width_pct_val < 50
        
        if position == 0:
            # Long conditions: TRIX turns positive + volume spike + low vol regime + above 1w EMA50
            if trix_val > 0 and trix_val > trix_aligned[i-1] and vol_spike and low_vol and price > ema50_1w:
                signals[i] = 0.25
                position = 1
            # Short conditions: TRIX turns negative + volume spike + low vol regime + below 1w EMA50
            elif trix_val < 0 and trix_val < trix_aligned[i-1] and vol_spike and low_vol and price < ema50_1w:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: TRIX momentum fades or volatility expands
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when TRIX turns negative or volatility expands (BB width > 80th percentile)
                if trix_val < 0 or bb_width_pct_val > 80:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when TRIX turns positive or volatility expands
                if trix_val > 0 or bb_width_pct_val > 80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_TRIX_Volume_Spike_Momentum"
timeframe = "1d"
leverage = 1.0