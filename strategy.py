#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI mean reversion with 4h EMA50 trend filter and volume confirmation
# Long when RSI < 30 (oversold) + price > 4h EMA50 (uptrend) + volume spike
# Short when RSI > 70 (overbought) + price < 4h EMA50 (downtrend) + volume spike
# Exit when RSI returns to neutral zone (40-60) or trend reverses
# Uses RSI for mean reversion in ranging markets, EMA50 for trend filter to avoid counter-trend trades
# Volume spike filters for institutional participation
# Designed for 15-30 trades/year with edge in both bull (buy dips) and bear (sell rallies)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate RSI(14) on 1h closes
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to alpha = 1/14)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (8:00-20:00 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(vol_ma_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_val = ema_50_aligned[i]
        rsi_val = rsi[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_spike = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: RSI oversold + uptrend + volume spike
            if rsi_val < 30 and price > ema_val and vol_spike:
                signals[i] = 0.20
                position = 1
            # Short conditions: RSI overbought + downtrend + volume spike
            elif rsi_val > 70 and price < ema_val and vol_spike:
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit conditions: RSI returns to neutral or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when RSI >= 40 (exiting oversold) or trend turns down
                if rsi_val >= 40 or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when RSI <= 60 (exiting overbought) or trend turns up
                if rsi_val <= 60 or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_RSI_MeanReversion_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0