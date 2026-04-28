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
    
    # Get daily data once for calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get weekly data once for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily high/low/close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range for pivot calculations
    daily_range = high_1d - low_1d
    
    # Weekly high/low/close for calculations
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range for pivot calculations
    weekly_range = high_1w - low_1w
    
    # Calculate daily ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily ATR ratio (ATR/ATR_MA) for volatility regime filter
    atr_ma = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = tr / (atr_ma + 1e-10)  # Avoid division by zero
    
    # Calculate daily RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate weekly EMA21 for trend
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate weekly RSI for trend confirmation
    delta_w = np.diff(close_1w, prepend=close_1w[0])
    gain_w = np.where(delta_w > 0, delta_w, 0)
    loss_w = np.where(delta_w < 0, -delta_w, 0)
    avg_gain_w = pd.Series(gain_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_w = pd.Series(loss_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_w = avg_gain_w / (avg_loss_w + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs_w))
    
    # Camarilla pivot levels (based on previous day)
    camarilla_r1 = close_1d + daily_range * 1.1 / 12
    camarilla_s1 = close_1d - daily_range * 1.1 / 12
    camarilla_r2 = close_1d + daily_range * 1.1 / 6
    camarilla_s2 = close_1d - daily_range * 1.1 / 6
    camarilla_r3 = close_1d + daily_range * 1.1 / 4
    camarilla_s3 = close_1d - daily_range * 1.1 / 4
    camarilla_r4 = close_1d + daily_range * 1.1 / 2
    camarilla_s4 = close_1d - daily_range * 1.1 / 2
    
    # Align levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Volume filter: above average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr_ratio[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when volatility is elevated (ATR ratio > 1.2)
        vol_filter = atr_ratio[i] > 1.2
        
        # Volume filter: above average volume
        vol_filter_vol = volume[i] > vol_ma[i]
        
        # Weekly trend filter: price above/below weekly EMA21 and RSI > 50 for uptrend, < 50 for downtrend
        price_above_weekly_ema = close[i] > ema_21_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_21_1w_aligned[i]
        weekly_uptrend = rsi_1w_aligned[i] > 50
        weekly_downtrend = rsi_1w_aligned[i] < 50
        
        # Entry conditions: 
        # Long: price breaks above S3 with volume and volatility, weekly uptrend
        # Short: price breaks below R3 with volume and volatility, weekly downtrend
        long_entry = (close[i] > s3_aligned[i]) and vol_filter and vol_filter_vol and weekly_uptrend
        short_entry = (close[i] < r3_aligned[i]) and vol_filter and vol_filter_vol and weekly_downtrend
        
        # Exit conditions: price returns to opposite R3/S3 levels or weekly trend reversal
        long_exit = (close[i] < r3_aligned[i]) or (not weekly_uptrend)
        short_exit = (close[i] > s3_aligned[i]) or (not weekly_downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Camarilla_S3R3_VolatilityFilter_EMA21"
timeframe = "1d"
leverage = 1.0