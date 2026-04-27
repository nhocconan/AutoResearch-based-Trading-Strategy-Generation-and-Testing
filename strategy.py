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
    
    # Get daily data for higher timeframe context (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily EMA(34) for trend direction
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily RSI(14) for momentum
    delta_1d = pd.Series(close_1d).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0)
    loss_1d = -delta_1d.where(delta_1d < 0, 0)
    avg_gain_1d = gain_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1d = loss_1d.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1d = avg_gain_1d / avg_loss_1d.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_values = rsi_1d.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # Calculate 12h Bollinger Bands for volatility regime
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    sma_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).mean()
    std_20_12h = pd.Series(close_12h).rolling(window=20, min_periods=20).std()
    upper_bb_12h = sma_20_12h + (std_20_12h * 2)
    lower_bb_12h = sma_20_12h - (std_20_12h * 2)
    bb_width_12h = (upper_bb_12h - lower_bb_12h) / sma_20_12h
    bb_width_aligned = align_htf_to_ltf(prices, df_12h, bb_width_12h.values)
    
    # Precompute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(bb_width_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during active hours
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Momentum filter: RSI in neutral zone (40-60) to avoid extremes
        rsi_momentum = (rsi_1d_aligned[i] >= 40) and (rsi_1d_aligned[i] <= 60)
        
        # Volatility filter: Bollinger Band width above median (avoid low volatility)
        vol_median = np.nanmedian(bb_width_aligned[:i+1])
        vol_filter = bb_width_aligned[i] > vol_median if not np.isnan(vol_median) else False
        
        # Entry conditions: trend + momentum + volatility
        long_condition = price_above_ema and rsi_momentum and vol_filter
        short_condition = price_below_ema and rsi_momentum and vol_filter
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: trend reversal or momentum extreme
        elif position == 1 and (not price_above_ema or rsi_1d_aligned[i] > 70):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (not price_below_ema or rsi_1d_aligned[i] < 30):
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

name = "1d_EMA34_RSI14_BBWidthFilter"
timeframe = "1d"
leverage = 1.0