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
    
    # Get daily data once for context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data once for trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily high/low/close for calculations
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily range for pivot calculations
    daily_range = high_1d - low_1d
    
    # Weekly high/low/close
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range for pivot calculations
    weekly_range = high_1w - low_1w
    
    # Calculate EMA50 on weekly close for trend
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate RSI on weekly close for momentum
    delta = pd.Series(close_1w).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = rsi_1w.values
    
    # Align indicators to daily timeframe
    weekly_range_aligned = align_htf_to_ltf(prices, df_1w, weekly_range)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate daily ATR for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Calculate daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate volume moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Hour filter: 8-20 UTC (most active trading hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_range_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(rsi_1w_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
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
        
        # Volatility filter: ATR > 0.5 * ATR MA (avoid low volatility)
        atr_ma = pd.Series(atr_14_aligned).rolling(window=20, min_periods=20).mean().values
        vol_filter = atr_14_aligned[i] > 0.5 * atr_ma[i]
        
        # Volume filter: above average volume
        vol_filter = vol_filter and (volume[i] > vol_ma[i])
        
        # Trend filter: price above/below daily EMA200
        price_above_ema200 = close[i] > ema_200_1d_aligned[i]
        price_below_ema200 = close[i] < ema_200_1d_aligned[i]
        
        # Weekly trend filter: price above/below weekly EMA50
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        # Weekly momentum filter: RSI not extreme
        rsi_not_overbought = rsi_1w_aligned[i] < 70
        rsi_not_oversold = rsi_1w_aligned[i] > 30
        
        # Calculate dynamic pivot levels based on weekly range
        # Upper pivot: previous day close + 0.6 * weekly range
        # Lower pivot: previous day close - 0.6 * weekly range
        upper_pivot = close_1d[i-1] + 0.6 * weekly_range_aligned[i]
        lower_pivot = close_1d[i-1] - 0.6 * weekly_range_aligned[i]
        
        # Entry conditions: 
        # Long: price breaks above upper pivot with volume, uptrend, and reasonable RSI
        # Short: price breaks below lower pivot with volume, downtrend, and reasonable RSI
        long_entry = (close[i] > upper_pivot) and price_above_ema200 and price_above_weekly_ema and vol_filter and rsi_not_overbought
        short_entry = (close[i] < lower_pivot) and price_below_ema200 and price_below_weekly_ema and vol_filter and rsi_not_oversold
        
        # Exit conditions: price returns to opposite pivot or trend reversal
        long_exit = (close[i] < lower_pivot) or (not price_above_ema200) or (not price_above_weekly_ema)
        short_exit = (close[i] > upper_pivot) or (not price_below_ema200) or (not price_below_weekly_ema)
        
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

name = "1d_WeeklyPivot_EMA200_RSI_Filter"
timeframe = "1d"
leverage = 1.0