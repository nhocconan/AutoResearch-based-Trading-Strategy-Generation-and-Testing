#2025-06-14T11:54:49.068Z
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
    
    # Get weekly data for higher timeframe context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly EMA(8) for trend direction
    ema_8_1w = pd.Series(close_1w).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_8_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_8_1w)
    
    # Calculate weekly ATR(10) for volatility filter
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    atr_10_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_10_1w)
    
    # Calculate weekly Bollinger Bands for range detection
    sma_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    std_20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).std().values
    upper_bb_1w = sma_20_1w + 2 * std_20_1w
    lower_bb_1w = sma_20_1w - 2 * std_20_1w
    upper_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_bb_1w)
    lower_bb_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_bb_1w)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_8_1w_aligned[i]) or 
            np.isnan(atr_10_1w_aligned[i]) or
            np.isnan(upper_bb_1w_aligned[i]) or
            np.isnan(lower_bb_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA8
        price_above_ema = close[i] > ema_8_1w_aligned[i]
        price_below_ema = close[i] < ema_8_1w_aligned[i]
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr_10_1w_aligned[i] > 0 and atr_10_1w_aligned[i] < np.median(atr_10_1w_aligned[:i+1]) * 3
        
        # Range filter: price outside weekly Bollinger Bands (breakout condition)
        price_above_upper = close[i] > upper_bb_1w_aligned[i]
        price_below_lower = close[i] < lower_bb_1w_aligned[i]
        
        # Volume filter: above average volume
        vol_ma_10_1w = pd.Series(volume_1w).rolling(window=10, min_periods=10).mean().values
        vol_ma_10_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_10_1w)
        if np.isnan(vol_ma_10_1w_aligned[i]):
            signals[i] = 0.0
            continue
        vol_spike = volume[i] > vol_ma_10_1w_aligned[i]
        
        # Long conditions: bullish breakout + volatility filter + volume spike
        long_condition = (price_above_upper and vol_filter and vol_spike)
        
        # Short conditions: bearish breakdown + volatility filter + volume spike
        short_condition = (price_below_lower and vol_filter and vol_spike)
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal
        elif position == 1 and not price_above_ema:
            signals[i] = 0.0
            position = 0
        elif position == -1 and not price_below_ema:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyEMA8_BBBreakout_VolumeFilter_Session"
timeframe = "1d"
leverage = 1.0