#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h VWAP + RSI(14) momentum with daily trend filter (EMA34)
# Uses daily EMA34 to determine long-term trend direction (bull/bear)
# In uptrend: long when price > VWAP and RSI > 50
# In downtrend: short when price < VWAP and RSI < 50
# VWAP resets daily, providing intraday mean reversion edge
# RSI filters for momentum confirmation
# Designed to work in both bull and bear markets by following higher timeframe trend
# Targets 20-40 trades/year with disciplined entries

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA34 for trend filter
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate VWAP (resets daily)
    typical_price = (prices['high'] + prices['low'] + prices['close']) / 3
    price_volume = typical_price * prices['volume']
    cum_pv = price_volume.cumsum()
    cum_vol = prices['volume'].cumsum()
    vwap = cum_pv / cum_vol
    
    # Handle first bar and daily reset
    vwap_array = vwap.values
    # Reset VWAP at daily boundaries (when date changes)
    dates = pd.to_datetime(prices['open_time']).date
    date_changes = np.concatenate([[True], dates[1:] != dates[:-1]])
    for i in range(1, len(vwap_array)):
        if date_changes[i]:
            cum_pv.iloc[i] = price_volume.iloc[i]
            cum_vol.iloc[i] = prices['volume'].iloc[i]
            vwap_array[i] = price_volume.iloc[i] / prices['volume'].iloc[i]
        else:
            cum_pv.iloc[i] = cum_pv.iloc[i-1] + price_volume.iloc[i]
            cum_vol.iloc[i] = cum_vol.iloc[i-1] + prices['volume'].iloc[i]
            vwap_array[i] = cum_pv.iloc[i] / cum_vol.iloc[i]
    
    # Calculate RSI(14)
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / avg_loss
    rs = np.where(avg_loss == 0, 100, rs)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(vwap_array[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vwap_val = vwap_array[i]
        ema_trend = ema_34_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Enter long in uptrend when price above VWAP and bullish momentum
            if price > ema_trend and price > vwap_val and rsi_val > 50:
                signals[i] = 0.25
                position = 1
            # Enter short in downtrend when price below VWAP and bearish momentum
            elif price < ema_trend and price < vwap_val and rsi_val < 50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price crosses below VWAP or RSI turns bearish
                if price < vwap_val or rsi_val < 50:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price crosses above VWAP or RSI turns bullish
                if price > vwap_val or rsi_val > 50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_VWAP_RSI_TrendFilter"
timeframe = "4h"
leverage = 1.0