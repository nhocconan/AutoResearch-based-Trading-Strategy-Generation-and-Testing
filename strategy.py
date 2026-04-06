#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h trend (EMA21) and 1d trend (EMA50) for direction,
# with 1h momentum (RSI14) for entry timing. Only trade during 08-20 UTC to avoid
# low-volume periods. Uses 1h Donchian breakout (20-period) with volume confirmation.
# Designed for 60-150 total trades over 4 years (15-37/year) to minimize fee drag.

name = "1h_4h_1d_trend_rsi_donchian_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA21 for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d EMA50 for higher timeframe trend
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1h RSI14 for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1h Donchian channel (20-period) for breakouts
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > vol_ma * 1.3
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if any data not available
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(rsi[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check if we're in trading session
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0  # Close position outside session
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend alignment: both 4h and 1d must agree
        long_trend = (close[i] > ema_4h_aligned[i]) and (close[i] > ema_1d_aligned[i])
        short_trend = (close[i] < ema_4h_aligned[i]) and (close[i] < ema_1d_aligned[i])
        
        # Check exits and stoploss
        if position == 1:  # long position
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            # Exit: trend reversal, stoploss, or RSI overbought
            if (not long_trend or 
                close[i] < stop_loss_level or 
                rsi[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            # Exit: trend reversal, stoploss, or RSI oversold
            if (not short_trend or 
                close[i] > stop_loss_level or 
                rsi[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume confirmation and trend alignment
            if volume_filter[i]:
                # Long entry: price breaks above Donchian high with uptrend and RSI not overbought
                if (long_trend and 
                    close[i] > donchian_high[i] and 
                    rsi[i] < 70):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short entry: price breaks below Donchian low with downtrend and RSI not oversold
                elif (short_trend and 
                      close[i] < donchian_low[i] and 
                      rsi[i] > 30):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals