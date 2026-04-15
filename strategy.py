#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h mean reversion with 4h trend filter and 1d regime filter
# Uses RSI(14) for mean reversion signals (long <30, short >70)
# Filters by 4h ADX > 25 for trending markets (avoid chop)
# Uses 1d Bollinger Bands width percentile to detect regime (narrow = range, wide = trend)
# In ranging markets (BBW < 50th percentile): mean reversion
# In trending markets (BBW >= 50th percentile): follow 4h trend
# Designed to work in both bull (trend following) and bear (mean reversion in ranges)
# Discrete sizing (0.20) to limit trade frequency and control drawdown

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h ADX for trend filter
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_4h[1:] - high_4h[:-1]) > (low_4h[:-1] - low_4h[1:]), 
                       np.maximum(high_4h[1:] - high_4h[:-1], 0), 0)
    minus_dm = np.where((low_4h[:-1] - low_4h[1:]) > (high_4h[1:] - high_4h[:-1]), 
                        np.maximum(low_4h[:-1] - low_4h[1:], 0), 0)
    tr = np.maximum(high_4h[1:] - low_4h[1:], 
                    np.absolute(high_4h[1:] - close_4h[:-1]), 
                    np.absolute(low_4h[1:] - close_4h[:-1]))
    
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    atr_4h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_4h = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_4h
    minus_di_4h = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_4h
    dx_4h = 100 * np.abs(plus_di_4h - minus_di_4h) / (plus_di_4h + minus_di_4h)
    adx_4h = pd.Series(dx_4h).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 4h ADX and DI to 1h
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    plus_di_4h_aligned = align_htf_to_ltf(prices, df_4h, plus_di_4h)
    minus_di_4h_aligned = align_htf_to_ltf(prices, df_4h, minus_di_4h)
    
    # 1d Bollinger Bands for regime detection
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Bollinger Bands (20, 2)
    sma_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20  # Normalized width
    
    # Calculate percentile of BB width (lookback 50 days)
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).apply(
        lambda x: np.percentile(x, 50) if len(x) >= 10 else np.nan, raw=True
    ).values
    
    # Align BB width percentile to 1h
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # 1h RSI for mean reversion signals
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    # Handle first value
    rsi = np.concatenate([[50], rsi])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(adx_4h_aligned[i]) or np.isnan(plus_di_4h_aligned[i]) or 
            np.isnan(minus_di_4h_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or
            np.isnan(rsi[i]) or not in_session[i]):
            continue
        
        # Determine market regime: ranging (BBW < 50th percentile) or trending (BBW >= 50th)
        is_ranging = bb_width_percentile_aligned[i] < 50.0
        
        if is_ranging:
            # Ranging market: mean reversion
            # Long: RSI < 30 (oversold)
            # Short: RSI > 70 (overbought)
            if rsi[i] < 30:
                signals[i] = 0.20
            elif rsi[i] > 70:
                signals[i] = -0.20
            else:
                signals[i] = 0.0  # Exit mean reversion position
        else:
            # Trending market: follow 4h trend
            # Long: +DI > -DI
            # Short: -DI > +DI
            if plus_di_4h_aligned[i] > minus_di_4h_aligned[i]:
                signals[i] = 0.20
            elif minus_di_4h_aligned[i] > plus_di_4h_aligned[i]:
                signals[i] = -0.20
            else:
                signals[i] = 0.0  # Exit trend position
    
    return signals

name = "1h_RSI_MeanReversion_4hADX_1dBBRegime"
timeframe = "1h"
leverage = 1.0