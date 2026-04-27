# 4h_ChoppinessIndex_VolatilityBreakout_Filter
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4-hour data for Choppiness Index and volatility
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour ATR(14) for volatility
    tr1 = df_4h['high'] - df_4h['low']
    tr2 = np.abs(df_4h['high'] - df_4h['close'].shift(1))
    tr3 = np.abs(df_4h['low'] - df_4h['close'].shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-hour Choppiness Index(14)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = df_4h['high'].rolling(window=14, min_periods=14).max().values
    ll = df_4h['low'].rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    
    # Calculate 4-hour volume MA(20)
    vol_4h = df_4h['volume'].values
    vol_ma_20_4h = pd.Series(vol_4h).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    chop_aligned = align_htf_to_ltf(prices, df_4h, chop)
    vol_ma_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_ma_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need ATR, chop, volume MA, and 1d EMA
    start_idx = max(14, 14, 20, 50)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        atr_val = atr_aligned[i]
        chop_val = chop_aligned[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_4h_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Volatility filter: ATR > 1.2x 4h average (normal to high volatility)
        vol_filter = atr_val > 1.2 * np.nanmedian(atr_aligned[max(0, i-50):i+1])
        
        # Chop filter: chop > 61.8 (ranging market) for mean reversion
        chop_filter = chop_val > 61.8
        
        # Entry conditions: volatility breakout in ranging market with trend alignment
        if position == 0:
            # Long: price above trend + volatility + chop
            if close[i] > trend_1d and vol_filter and chop_filter:
                signals[i] = size
                position = 1
            # Short: price below trend + volatility + chop
            elif close[i] < trend_1d and vol_filter and chop_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: chop drops below 38.2 (trending) or price crosses trend
            if chop_val < 38.2 or close[i] < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: chop drops below 38.2 or price crosses trend
            if chop_val < 38.2 or close[i] > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_ChoppinessIndex_VolatilityBreakout_Filter"
timeframe = "4h"
leverage = 1.0