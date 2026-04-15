#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Bollinger Band width squeeze + RSI mean reversion + volume confirmation
# In ranging markets, Bollinger Band width narrows (squeeze). When price moves outside
# Bollinger Bands with RSI extreme and volume spike, it often signals a mean-reversion bounce.
# Works in both bull and bear markets by fading extremes. Target: 50-100 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for trend filter (avoid trading against weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Bollinger Bands on daily
    bb_length = 20
    bb_mult = 2.0
    
    # Basis (SMA)
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    # Standard deviation
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    # Upper and lower bands
    upper = basis + dev
    lower = basis - dev
    # Bollinger Band width (normalized)
    bb_width = (upper - lower) / basis
    
    # Calculate 14-period RSI
    rsi_length = 14
    delta = pd.Series(close).diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=rsi_length, min_periods=rsi_length).mean().values
    avg_loss = pd.Series(loss).rolling(window=rsi_length, min_periods=rsi_length).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to daily timeframe
    bb_width_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), bb_width)
    rsi_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), rsi)
    basis_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), basis)
    upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), upper)
    lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lower)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25%)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(bb_width_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(basis_aligned[i]) or np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            continue
        
        # Bollinger Band width squeeze threshold (lower 20% of recent values)
        bb_width_lookback = bb_width_aligned[max(0, i-50):i+1]
        if len(bb_width_lookback) < 10:
            squeeze_condition = False
        else:
            bb_width_percentile = (bb_width_aligned[i] <= np.percentile(bb_width_lookback, 20))
        
        # RSI extreme conditions
        rsi_overbought = rsi_aligned[i] > 70
        rsi_oversold = rsi_aligned[i] < 30
        
        # Price outside Bollinger Bands
        price_above_upper = close[i] > upper_aligned[i]
        price_below_lower = close[i] < lower_aligned[i]
        
        # Volume confirmation (volume > 1.5x average of last 20 periods)
        vol_ma = np.mean(volume[max(0, i-20):i+1]) if i >= 20 else np.mean(volume[:i+1])
        volume_spike = volume[i] > 1.5 * vol_ma
        
        # Weekly trend filter: only trade against the weekly trend
        # In uptrend, look for short opportunities; in downtrend, look for long opportunities
        weekly_uptrend = close[i] > ema_50_1w_aligned[i]
        
        # Long entry: price below lower band + RSI oversold + squeeze + volume spike + weekly downtrend
        if (price_below_lower and rsi_oversold and bb_width_percentile and 
            volume_spike and not weekly_uptrend and position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price above upper band + RSI overbought + squeeze + volume spike + weekly uptrend
        elif (price_above_upper and rsi_overbought and bb_width_percentile and 
              volume_spike and weekly_uptrend and position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price returns to Bollinger Band basis or RSI returns to neutral
        elif position == 1 and (close[i] >= basis_aligned[i] or rsi_aligned[i] >= 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= basis_aligned[i] or rsi_aligned[i] <= 50):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "1d_BollingerSqueeze_RSI_MeanReversion"
timeframe = "1d"
leverage = 1.0