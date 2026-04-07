#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Bollinger Band squeeze breakout with 12-hour volume confirmation and daily RSI trend filter
# Long when price breaks above upper BB(20,2) + BB width < 5th percentile of 50-period + volume > 1.5x 20-period avg + daily RSI > 50
# Short when price breaks below lower BB(20,2) + BB width < 5th percentile + volume > 1.5x 20-period avg + daily RSI < 50
# Exit when price crosses middle Bollinger Band (20-period SMA)
# Stoploss at 2.5 * ATR(20)
# Position size: 0.25 (25% of capital)
# Target: 75-150 total trades over 4 years (19-38/year)

name = "6h_bb_squeeze_12h_vol_1d_rsi_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 1-day data for RSI trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 20-period Bollinger Bands
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2 * std_20
    lower_bb = sma_20 - 2 * std_20
    bb_width = (upper_bb - lower_bb) / sma_20 * 100  # as percentage
    
    # BB width percentile (5th percentile of 50-period lookback)
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == 50 else np.nan, raw=False
    ).values
    
    # 12-hour volume average (20-period)
    volume_12h = df_12h['volume'].values
    volume_12h_s = pd.Series(volume_12h)
    volume_ma = volume_12h_s.rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma)
    
    # 1-day RSI (14-period)
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain)
    loss_s = pd.Series(loss)
    avg_gain = gain_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = loss_s.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # ATR(20) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(upper_bb[i]) or np.isnan(lower_bb[i]) or 
            np.isnan(bb_width_percentile[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.5 * ATR
            if close[i] < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses below middle Bollinger Band (SMA20)
            elif close[i] < sma_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Stoploss: 2.5 * ATR
            if close[i] > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: price crosses above middle Bollinger Band (SMA20)
            elif close[i] > sma_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Bollinger Band breakout with squeeze, volume confirmation and RSI filter
            # Squeeze filter: BB width below 5th percentile of 50-period
            squeeze_filter = bb_width_percentile[i] < 5
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma_aligned[i]
            # Trend filter: daily RSI > 50 for long, < 50 for short
            
            # Long: price breaks above upper BB + squeeze + volume filter + RSI > 50
            if close[i] > upper_bb[i] and squeeze_filter and volume_filter and rsi_aligned[i] > 50:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below lower BB + squeeze + volume filter + RSI < 50
            elif close[i] < lower_bb[i] and squeeze_filter and volume_filter and rsi_aligned[i] < 50:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals