#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Volume-Weighted MACD with 1d Regime Filter
# - MACD(12,26,9) on volume-weighted price (VWP) for momentum
# - 1d ADX(14) > 25 for trending regime filter (avoid whipsaws in ranging markets)
# - Volume confirmation: current 12h volume > 1.3x 20-period 1d average volume
# - Long: MACD line > signal line AND VWP > VWP EMA(50) AND ADX > 25 AND volume confirmation
# - Short: MACD line < signal line AND VWP < VWP EMA(50) AND ADX > 25 AND volume confirmation
# - Uses discrete position sizing: ±0.25 to limit drawdown and reduce fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) to stay within fee drag limits
# - Volume-weighted price reduces noise from low-volume spikes
# - ADX regime filter ensures we only trade in trending conditions
# - Works in both bull (strong upward momentum) and bear (strong downward momentum)

name = "12h_vwap_macd_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load 1d data ONCE before loop for regime and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return signals
    
    # Pre-compute 1d ADX for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 1d average volume for confirmation
    volume_1d = df_1d['volume'].values
    volume_sma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_sma_20_aligned = align_htf_to_ltf(prices, df_1d, volume_sma_20_1d)
    
    # Pre-compute Volume-Weighted Price (VWP) on 12h timeframe
    typical_price = (high + low + close) / 3
    vwp = np.sum(typical_price * volume) / np.sum(volume)  # Will compute properly below
    # Instead compute cumulative VWP-like indicator: VWMA
    vwma_numerator = pd.Series(typical_price * volume).rolling(window=21, min_periods=21).sum().values
    vwma_denominator = pd.Series(volume).rolling(window=21, min_periods=21).sum().values
    vwma = vwma_numerator / vwma_denominator
    
    # Pre-compute MACD on VWMA
    vwma_series = pd.Series(vwma)
    ema12 = vwma_series.ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = vwma_series.ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_histogram = macd_line - signal_line
    
    # Pre-compute VWMA EMA(50) for trend filter
    vwma_ema50 = pd.Series(vwma).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    for i in range(100, n):  # Start after 100-bar warmup
        # Skip if any required data is invalid
        if (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or
            np.isnan(vwma[i]) or np.isnan(vwma_ema50[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(volume_sma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price data
        volume_current = volume[i]
        
        # MACD conditions
        macd_bullish = macd_line[i] > signal_line[i]
        macd_bearish = macd_line[i] < signal_line[i]
        
        # Trend filter: VWMA above/below its EMA50
        vwma_uptrend = vwma[i] > vwma_ema50[i]
        vwma_downtrend = vwma[i] < vwma_ema50[i]
        
        # Regime filter: ADX > 25 indicates trending market
        trending_regime = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.3x 20-period 1d average
        vol_confirm = volume_current > 1.3 * volume_sma_20_aligned[i]
        
        # Entry conditions
        enter_long = False
        enter_short = False
        
        # Long: bullish MACD + uptrend VWMA + trending regime + volume confirmation
        if macd_bullish and vwma_uptrend and trending_regime and vol_confirm:
            enter_long = True
        
        # Short: bearish MACD + downtrend VWMA + trending regime + volume confirmation
        if macd_bearish and vwma_downtrend and trending_regime and vol_confirm:
            enter_short = True
        
        # Exit conditions: reverse MACD or loss of trend
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long if MACD turns bearish OR VWMA loses uptrend
            exit_long = (not macd_bullish) or (not vwma_uptrend)
        elif position == -1:
            # Exit short if MACD turns bullish OR VWMA loses downtrend
            exit_short = (not macd_bearish) or (not vwma_downtrend)
        
        # Trading logic
        if enter_long and position != 1:
            position = 1
            signals[i] = 0.25
        elif enter_short and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals