#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour momentum strategy using 4-hour EMA(50) trend filter and RSI(14) volume spikes
# Long when price > 4h EMA50 + RSI(14) > 55 + volume > 1.5x 20-period average
# Short when price < 4h EMA50 + RSI(14) < 45 + volume > 1.5x 20-period average
# Exit when RSI crosses back to 50 (mean reversion)
# Stoploss at 2.0 * ATR(14)
# Position size: 0.20 (20% of capital)
# Uses 4h EMA for trend direction, 1h RSI/volume for entry timing
# Target: 75-150 total trades over 4 years (19-38/year)

name = "1h_momentum_4h_ema50_rsi_vol_spike_v1"
timeframe = "1h"
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
    
    # 4-hour data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4-hour EMA(50)
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) calculation
    def calculate_rsi(close_prices, period=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain_ema = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        loss_ema = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        rs = gain_ema / (loss_ema + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Stoploss: 2.0 * ATR
            if close[i] < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses below 50 (mean reversion)
            elif rsi[i] < 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Stoploss: 2.0 * ATR
            if close[i] > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # Exit: RSI crosses above 50 (mean reversion)
            elif rsi[i] > 50:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: momentum with volume confirmation
            # Volume filter: volume > 1.5x 20-period average
            volume_filter = volume[i] > 1.5 * volume_ma[i]
            # Trend filter: price vs 4h EMA50
            price_above_ema = close[i] > ema_50_4h_aligned[i]
            price_below_ema = close[i] < ema_50_4h_aligned[i]
            
            # Long: price above 4h EMA50 + RSI > 55 + volume filter
            if price_above_ema and rsi[i] > 55 and volume_filter:
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
            # Short: price below 4h EMA50 + RSI < 45 + volume filter
            elif price_below_ema and rsi[i] < 45 and volume_filter:
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
    
    return signals