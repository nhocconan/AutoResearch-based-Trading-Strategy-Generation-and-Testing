#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour RSI divergence with 1-day volume confirmation and 1-week ADX trend filter.
# Looks for bearish/bullish RSI divergences (price makes new high/low but RSI does not)
# to anticipate reversals. Volume spike confirms conviction. 1-week ADX > 25 ensures
# we only trade in strong trending markets, avoiding whipsaws in ranging conditions.
# Works in both bull and bear markets by catching reversal points within trends.
# Target: 20-50 trades/year (80-200 total over 4 years) to balance opportunity and cost.
name = "4h_RSIDivergence_1dVolume_1wADX"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for RSI calculation
    df_4h = get_htf_data(prices, '4h')
    close_4h = pd.Series(df_4h['close'].values)
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    
    # Calculate RSI(14) on 4h data
    delta = close_4h.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align RSI to lower timeframe (4h is already our base, but we use it for alignment consistency)
    rsi_aligned = align_htf_to_ltf(prices, df_4h, rsi_values)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma_20 = vol_1d.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = align_htf_to_ltf(prices, df_1d, vol_1d > (2.0 * vol_ma_20))
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = pd.Series(df_1w['high'].values)
    low_1w = pd.Series(df_1w['low'].values)
    close_1w = pd.Series(df_1w['close'].values)
    
    # Calculate ADX(14) on 1w data
    tr1 = high_1w - low_1w
    tr2 = abs(high_1w - close_1w.shift(1))
    tr3 = abs(low_1w - close_1w.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1w = tr.rolling(window=14, min_periods=14).mean()
    
    up_move = high_1w.diff()
    down_move = low_1w.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / atr_1w)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1w = dx.ewm(alpha=1/14, adjust=False).mean().values
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Calculate price extrema for divergence detection
    # Look for new highs/lows over 5-period window
    highest_high = pd.Series(high).rolling(window=5, min_periods=5).max().values
    lowest_low = pd.Series(low).rolling(window=5, min_periods=5).min().values
    
    # Align price extrema
    highest_high_aligned = align_htf_to_ltf(prices, df_4h, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_4h, lowest_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi_aligned[i]) or np.isnan(volume_spike_1d[i]) or 
            np.isnan(adx_1w_aligned[i]) or np.isnan(highest_high_aligned[i]) or
            np.isnan(lowest_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        rsi_val = rsi_aligned[i]
        price = close[i]
        
        # Detect bearish divergence: price makes new high but RSI does not
        bearish_div = (price >= highest_high_aligned[i]) and (rsi_val < rsi_aligned[i-1])
        
        # Detect bullish divergence: price makes new low but RSI does not
        bullish_div = (price <= lowest_low_aligned[i]) and (rsi_val > rsi_aligned[i-1])
        
        # Volume spike confirmation
        vol_confirm = volume_spike_1d[i]
        
        # Strong trend filter
        strong_trend = adx_1w_aligned[i] > 25
        
        if position == 0:
            # Long: Bullish divergence + volume confirmation + strong trend
            if bullish_div and vol_confirm and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Bearish divergence + volume confirmation + strong trend
            elif bearish_div and vol_confirm and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish divergence OR RSI overbought (>70) OR trend weakens
            if bearish_div or rsi_val > 70 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish divergence OR RSI oversold (<30) OR trend weakens
            if bullish_div or rsi_val < 30 or adx_1w_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals