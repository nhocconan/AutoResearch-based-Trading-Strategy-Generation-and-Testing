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
    
    # Get daily data for higher timeframe context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d RSI(14) for overbought/oversold conditions
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h Bollinger Bands (20, 2) for mean reversion signals
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + (2 * bb_std)
    bb_lower = bb_middle - (2 * bb_std)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(bb_middle[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # Mean reversion signals: price at Bollinger Bands
        at_lower_band = close[i] <= bb_lower[i]
        at_upper_band = close[i] >= bb_upper[i]
        
        # RSI conditions: oversold (<30) or overbought (>70)
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Long conditions: price at lower BB + oversold RSI + above daily EMA + volume
        long_signal = at_lower_band and rsi_oversold and price_above_ema and volume_filter[i]
        # Short conditions: price at upper BB + overbought RSI + below daily EMA + volume
        short_signal = at_upper_band and rsi_overbought and price_below_ema and volume_filter[i]
        
        if long_signal:
            signals[i] = 0.20
            position = 1
        elif short_signal:
            signals[i] = -0.20
            position = -1
        # Exit conditions: price crosses back to middle Bollinger Band
        elif position == 1 and close[i] >= bb_middle[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] <= bb_middle[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_BollingerRSI_1dEMA50_MeanReversion"
timeframe = "1h"
leverage = 1.0